package template

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"io"
	"math"
	"slices"
	"strings"
	"sync"
	"text/template"
	"text/template/parse"

	"github.com/agnivade/levenshtein"
	"golang.org/x/exp/maps"

	"github.com/ollama/ollama/api"
)

//go:embed index.json
var indexBytes []byte

//go:embed *.gotmpl
//go:embed *.json
var templatesFS embed.FS

// templatesOnce ensures templates are loaded and parsed only once
var templatesOnce = sync.OnceValues(func() ([]*named, error) {
	var templates []*named
	if err := json.Unmarshal(indexBytes, &templates); err != nil {
		return nil, err
	}

	for _, t := range templates {
		bts, err := templatesFS.ReadFile(t.Name + ".gotmpl")
		if err != nil {
			return nil, err
		}

		// Normalize line endings to Unix style
		t.Bytes = bytes.ReplaceAll(bts, []byte("\r\n"), []byte("\n"))

		params, err := templatesFS.ReadFile(t.Name + ".json")
		if err != nil {
			// Missing parameters JSON is not fatal — continue
			continue
		}

		if err := json.Unmarshal(params, &t.Parameters); err != nil {
			return nil, err
		}
	}

	return templates, nil
})

// named represents a template with its metadata and parameters
type named struct {
	Name       string `json:"name"`
	Template   string `json:"template"`
	Bytes      []byte
	Parameters *struct {
		Stop []string `json:"stop"`
	}
}

// Reader returns an io.Reader for the raw template bytes
func (t named) Reader() io.Reader {
	return bytes.NewReader(t.Bytes)
}

// Named looks up the closest matching template by Levenshtein distance
func Named(s string) (*named, error) {
	templates, err := templatesOnce()
	if err != nil {
		return nil, err
	}

	var bestMatch *named
	bestScore := math.MaxInt

	for _, t := range templates {
		dist := levenshtein.ComputeDistance(s, t.Template)
		if dist < bestScore {
			bestScore = dist
			bestMatch = t
		}
	}

	if bestScore < 100 {
		return bestMatch, nil
	}

	return nil, errors.New("no matching template found")
}

// DefaultTemplate is a simple template that outputs the Prompt
var DefaultTemplate, _ = Parse("{{ .Prompt }}")

// Template wraps text/template.Template and stores the raw source string
type Template struct {
	*template.Template
	raw string
}

// response is a prebuilt template node representing {{ .Response }}
var response = parse.ActionNode{
	NodeType: parse.NodeAction,
	Pipe: &parse.PipeNode{
		NodeType: parse.NodePipe,
		Cmds: []*parse.CommandNode{
			{
				NodeType: parse.NodeCommand,
				Args: []parse.Node{
					&parse.FieldNode{
						NodeType: parse.NodeField,
						Ident:    []string{"Response"},
					},
				},
			},
		},
	},
}

// funcs defines template helper functions available within templates
var funcs = template.FuncMap{
	"json": func(v any) string {
		b, _ := json.Marshal(v)
		return string(b)
	},
}

// Parse creates a new Template from a string, adding {{ .Response }} if needed
func Parse(s string) (*Template, error) {
	tmpl := template.New("").Option("missingkey=zero").Funcs(funcs)

	tmpl, err := tmpl.Parse(s)
	if err != nil {
		return nil, err
	}

	t := Template{Template: tmpl, raw: s}

	// If the template doesn't use "messages" or "response", append {{ .Response }}
	vars := t.Vars()
	if !slices.Contains(vars, "messages") && !slices.Contains(vars, "response") {
		tmpl.Tree.Root.Nodes = append(tmpl.Tree.Root.Nodes, &response)
	}

	return &t, nil
}

// String returns the raw template source string
func (t *Template) String() string {
	return t.raw
}

// Vars returns a sorted list of all variable identifiers used in the template
func (t *Template) Vars() []string {
	var vars []string

	for _, tt := range t.Templates() {
		for _, n := range tt.Root.Nodes {
			vars = append(vars, Identifiers(n)...)
		}
	}

	set := make(map[string]struct{})
	for _, v := range vars {
		set[strings.ToLower(v)] = struct{}{}
	}

	vars = maps.Keys(set)
	slices.Sort(vars)
	return vars
}

// Values holds data passed to Execute when rendering a template
type Values struct {
	Messages []api.Message
	api.Tools
	Prompt string
	Suffix string

	forceLegacy bool // flag for legacy template compatibility testing
}

// Subtree returns a template containing the first node that matches the predicate fn
func (t *Template) Subtree(fn func(parse.Node) bool) *template.Template {
	var walk func(parse.Node) parse.Node

	walk = func(n parse.Node) parse.Node {
		if fn(n) {
			return n
		}

		switch node := n.(type) {
		case *parse.ListNode:
			for _, c := range node.Nodes {
				if res := walk(c); res != nil {
					return res
				}
			}
		case *parse.BranchNode:
			for _, sublist := range []*parse.ListNode{node.List, node.ElseList} {
				if sublist != nil {
					if res := walk(sublist); res != nil {
						return res
					}
				}
			}
		case *parse.IfNode:
			return walk(&node.BranchNode)
		case *parse.WithNode:
			return walk(&node.BranchNode)
		case *parse.RangeNode:
			return walk(&node.BranchNode)
		}

		return nil
	}

	if n := walk(t.Tree.Root); n != nil {
		tree := &parse.Tree{
			Root: &parse.ListNode{
				Nodes: []parse.Node{n},
			},
		}
		return template.New("").Funcs(funcs).ParseTree(tree)
	}

	return nil
}

// Execute renders the template with the provided Values, supporting legacy mode and various variable sets
func (t *Template) Execute(w io.Writer, v Values) error {
	system, messages := collate(v.Messages)

	// Shortcut for Prompt + Suffix templates
	if v.Prompt != "" && v.Suffix != "" {
		return t.Template.Execute(w, map[string]any{
			"Prompt":   v.Prompt,
			"Suffix":   v.Suffix,
			"Response": "",
		})
	}

	// If not legacy mode and template uses messages, pass them directly
	if !v.forceLegacy && slices.Contains(t.Vars(), "messages") {
		return t.Template.Execute(w, map[string]any{
			"System":   system,
			"Messages": messages,
			"Tools":    v.Tools,
			"Response": "",
		})
	}

	// Legacy rendering: execute template multiple times per message role
	system = ""
	var b bytes.Buffer
	var prompt, response string

	for _, m := range messages {
		execute := func() error {
			err := t.Template.Execute(&b, map[string]any{
				"System":   system,
				"Prompt":   prompt,
				"Response": response,
			})
			system, prompt, response = "", "", ""
			return err
		}

		switch m.Role {
		case "system":
			if prompt != "" || response != "" {
				if err := execute(); err != nil {
					return err
				}
			}
			system = m.Content

		case "user":
			if response != "" {
				if err := execute(); err != nil {
					return err
				}
			}
			prompt = m.Content

		case "assistant":
			response = m.Content
		}
	}

	// Remove {{ .Response }} node from the template and re-execute
	var cut bool
	nodes := deleteNode(t.Template.Root.Copy(), func(n parse.Node) bool {
		if field, ok := n.(*parse.FieldNode); ok && slices.Contains(field.Ident, "Response") {
			cut = true
			return false
		}
		return cut
	})

	tree := parse.Tree{Root: nodes.(*parse.ListNode)}
	if err := template.Must(template.New("").AddParseTree("", &tree)).Execute(&b, map[string]any{
		"System":   system,
		"Prompt":   prompt,
		"Response": response,
	}); err != nil {
		return err
	}

	_, err := io.Copy(w, &b)
	return err
}

// collate merges consecutive messages of the same role and collects system messages
// also mutates message content by appending image tags as needed
func collate(msgs []api.Message) (string, []*api.Message) {
	var system []string
	var collated []*api.Message

	for i := range msgs {
		msg := msgs[i]

		if msg.Role == "system" {
			system = append(system, msg.Content)
		}

		if len(collated) > 0 && collated[len(collated)-1].Role == msg.Role {
			collated[len(collated)-1].Content += "\n\n" + msg.Content
		} else {
			collated = append(collated, &msg)
		}
	}

	return strings.Join(system, "\n\n"), collated
}

// Identifiers recursively walks a parse.Node tree and returns all identifiers found
func Identifiers(n parse.Node) []string {
	switch node := n.(type) {
	case *parse.ListNode:
		var names []string
		for _, c := range node.Nodes {
			names = append(names, Identifiers(c)...)
		}
		return names

	case *parse.TemplateNode, *parse.ActionNode:
		return Identifiers(node.(interface{ PipeNode() parse.Node }).PipeNode())

	case *parse.BranchNode:
		names := Identifiers(node.Pipe)
		for _, list := range []*parse.ListNode{node.List, node.ElseList} {
			if list != nil {
				names = append(names, Identifiers(list)...)
			}
		}
		return names

	case *parse.IfNode, *parse.RangeNode, *parse.WithNode:
		return Identifiers(&node.(*parse.BranchNode).BranchNode)

	case *parse.PipeNode:
		var names []string
		for _, cmd := range node.Cmds {
			for _, arg := range cmd.Args {
				names = append(names, Identifiers(arg)...)
			}
		}
		return names

	case *parse.FieldNode, *parse.VariableNode:
		return node.Ident
	}

	return nil
}

// deleteNode walks a node tree and removes nodes matching the predicate fn
// used to remove the {{ .Response }} node from templates
func deleteNode(n parse.Node, fn func(parse.Node) bool) parse.Node {
	var walk func(parse.Node) parse.Node
	walk = func(node parse.Node) parse.Node {
		if fn(node) {
			return nil
		}

		switch t := node.(type) {
		case *parse.ListNode:
			var filtered []parse.Node
			for _, c := range t.Nodes {
				if nn := walk(c); nn != nil {
					filtered = append(filtered, nn)
				}
			}
			t.Nodes = filtered
			return t

		case *parse.IfNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.WithNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.RangeNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.BranchNode:
			t.List = walk(t.List).(*parse.ListNode)
			if t.ElseList != nil {
				t.ElseList = walk(t.ElseList).(*parse.ListNode)
			}
		case *parse.ActionNode:
			nn := walk(t.Pipe)
			if nn == nil {
				return nil
			}
			t.Pipe = nn.(*parse.PipeNode)
		case *parse.PipeNode:
			var cmds []*parse.CommandNode
			for _, c := range t.Cmds {
				var args []parse.Node
				for _, a := range c.Args {
					if nn := walk(a); nn != nil {
						args = append(args, nn)
					}
				}
				if len(args) == 0 {
					return nil
				}
				c.Args = args
				cmds = append(cmds, c)
			}
			if len(cmds) == 0 {
				return nil
			}
			t.Cmds = cmds
		}

		return node
	}

	return walk(n)
}
