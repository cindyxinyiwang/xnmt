from input import *
from vocab import RuleVocab, Rule
from collections import defaultdict, deque
import codecs
import argparse

def construct(out_file):
    ''' Construct merged bpe rules from bpe.out '''
    def split(s, rule_dict, merged_set):
        s = s.split(u'|||')
        assert len(s) == 3, s
        label, children, open_nonterms = s[0], s[1].split(), s[2].split()

        if label in rule_dict:
            merged_set.add(label)
            return rule_dict[label].copy()
        else:
            node = TreeNode(label, [])
        for c in children:
            if c in open_nonterms:
                if c in rule_dict:
                    merged_set.add(c)
                    node.add_child(rule_dict[c].copy())
                else:
                    n = TreeNode(c, [])
                    node.add_child(n)
            else:
                node.add_child(c)
        return node
    myfile = codecs.open(out_file, encoding='utf-8')
    label = 1
    rule_dict = {}
    merged_label = set()
    while True:
        line = myfile.readline()
        if not line: break
        par_node = split(line, rule_dict, merged_label)
        child_node = split(myfile.readline(), rule_dict, merged_label)
        myfile.readline()
        # find all child_rule parent in par_rule child
        # replace with child_rule children
        for c in par_node.frontir_nodes():
            if c.label == child_node.label:
                #c = child_node
                #print child_node.to_parse_string()
                c.children = []
                for i in child_node.children:
                    if hasattr(i, 'copy'):
                        c.children.append(i.copy())
                    else:
                        c.children.append(i)
        rule_dict[str(label)] = par_node
        label += 1

    for i, r in rule_dict.items():
        if i not in merged_label:
            print r.to_parse_string().encode('utf-8')

def get_rule(treenode):
    children, open_nonterms = [], []
    for c in treenode.children:
        if type(c) == str or type(c) == unicode:
            children.append(c)
        else:
            children.append(c.label)
            open_nonterms.append(c.label)
    return Rule(treenode.label, children, open_nonterms)

def str_to_rule(s):
    s = s.split('|||')
    assert len(s) == 3, s
    return Rule(s[0], s[1].split(), s[2].split())

def collect(treenode, rule_vocab, count_dict, conn_dict):
    init = []
    for c in treenode.children:
        if not hasattr(c, 'is_preterminal'):
            print 'root tree node is a preterminal. this should not happen'
            continue
        if not c.is_preterminal():
            init.append(c)
    visited = deque(init)
    while visited:
        cur = visited.popleft()
        cur_rule_id = rule_vocab.convert(get_rule(cur))
        for c in cur.children:
            if not hasattr(c, 'is_preterminal'):
                continue  # continue if c is terminal string
            if not c.is_preterminal():
                visited.append(c)
            c_rule_id = rule_vocab.convert(get_rule(c))
            count_dict[(cur_rule_id, c_rule_id)] += 1

def replace(treenode, par_rule, child_rule, label):
    init = []
    for c in treenode.children:
        if not hasattr(c, 'is_preterminal'):
            print 'root tree node is a preterminal. this should not happen'
            continue
        if not c.is_preterminal():
            init.append(c)
    visited = deque(init)
    label = str(label)
    while visited:
        cur = visited.popleft()
        cur_rule = get_rule(cur)

        for i, c in enumerate(cur.children):
            if not hasattr(c, 'is_preterminal'):
                continue  # ignore and continue if c is terminal string
            if not c.is_preterminal():
                visited.append(c)
            c_rule = get_rule(c)
            if cur_rule == par_rule and c_rule == child_rule:
                # merge this pair
                new_children = cur.children[:i] + c.children + cur.children[i+1:]
                for grand_c in c.children:
                    if hasattr(grand_c, 'children'):
                        visited.append(grand_c)
                for k in range(i+1, len(cur.children)):
                    if not hasattr(cur.children[k], 'is_preterminal'): continue
                    if not cur.children[k].is_preterminal():
                        visited.append(cur.children[k])
                cur.label = label
                cur.children = new_children

def BPE(tree_list, max_iter=10, root='ROOT'):
    label = 1
    for i in range(max_iter):
        # convert tree into rule count
        rule_vocab = RuleVocab()
        count_dict = defaultdict(int) # key: tuple, value: count
        conn_dict = defaultdict(list)  # key: tuple, value: list of connected tuple
        for tree in tree_list:
            collect(tree, rule_vocab, count_dict, conn_dict)
        rule_vocab.freeze()
        rule_vocab.set_unk(RuleVocab.UNK_STR)
        max_pair = max(count_dict, key=count_dict.get)
        print rule_vocab[max_pair[0]]
        print rule_vocab[max_pair[1]]
        print
        for tree in tree_list:
            replace(tree, rule_vocab[max_pair[0]], rule_vocab[max_pair[1]], label)
        label += 1

def test():
    orig_tree_file = "../examples/data/dev.parse.en"
    piece_file="../examples/data/dev.piece.en"
    tree_list = []

    tree_fp = codecs.open(orig_tree_file, 'r', encoding='utf-8')
    piece_fp = codecs.open(piece_file, 'r', encoding='utf-8')
    for line in tree_fp:
        piece = piece_fp.readline()
        t = Tree(parse_root(tokenize(line)), sent_piece=piece, binarize=True)
        #t = Tree(parse_root(tokenize(line)), binarize=False)
        tree_list.append(t.root)
    #for t in tree_list:
    #    print t.to_parse_string()

    BPE(tree_list, max_iter=4, root='ROOT')

    for t in tree_list:
        print t.to_parse_string()
        #print t.to_string(piece=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_file", required=True, type=str)
    parser.add_argument("--piece_file", type=str)
    parser.add_argument("--replace_file", type=str)
    parser.add_argument("--root", type=str, default='ROOT')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--construct", action='store_true')

    args = parser.parse_args()

    if args.construct:
        construct(args.replace_file)
        exit(0)
    tree_fp = codecs.open(args.tree_file, 'r', encoding='utf-8')
    tree_list = []
    if args.piece_file:
        piece_fp = codecs.open(args.piece_file, 'r', encoding='utf-8')
    for line in tree_fp:
        if args.piece_file:
            piece = piece_fp.readline()
            t = Tree(parse_root(tokenize(line)), sent_piece=piece, binarize=True)
        else:
            t = Tree(parse_root(tokenize(line)), binarize=True)
        tree_list.append(t.root.children[0])   # Tree adds an extra xxx node as parent to each root node
    #for t in tree_list:
    #    print t.to_parse_string()
    if args.replace_file:
        myfile = codecs.open(args.replace_file, encoding='utf-8')
        label = 1
        while True:
            line = myfile.readline()
            if not line: break
            par_rule = str_to_rule(line)
            child_rule = str_to_rule(myfile.readline())
            myfile.readline()
            for tree in tree_list:
                replace(tree, par_rule, child_rule, label)
            label += 1
    else:
        BPE(tree_list, max_iter=args.max_iter, root=args.root)

    with codecs.open(args.out_file, encoding='utf-8', mode='w') as out:
        for t in tree_list:
            out.write(u'{}\n'.format(t.to_parse_string()))