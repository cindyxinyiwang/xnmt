from input import *
from collections import defaultdict
import codecs
import argparse


def replace(tree_list, replace_struct, replaced_label):
    ''' replacce the rule with a label in a list of trees '''
    def _replace(tree, par, chi):
        child_list = [c.label if hasattr(c, 'label') else c for c in tree.children]
        #child_set = set(child_list)
        if tree.label == par and chi in child_list:
            # replace struct
            idx = child_list.index(chi)
            #if hasattr(child_list[idx], 'is_preterminal') and (not child_list[idx].is_preterminal()):
            new_children = []
            for i, child in enumerate(tree.children):
                if i == idx:
                    for grand_c in child.children:
                        #grand_c.label = u'{}_{}[{}]'.format(grand_c.label, replaced_label, child.label)
                        new_children.append(grand_c)
                else:
                    new_children.append(child)
            tree.label = replaced_label
            tree.children = new_children
        for c in tree.children:
            if hasattr(c, 'label'):
                _replace(c, par, chi)

    replace_struct = replace_struct.split(u'->')
    par, chi = replace_struct[0], replace_struct[1]
    for tree in tree_list:
        _replace(tree, par, chi)

def collect_structs(tree_list, root):
    ''' collect counts for tree structure '''
    def _collect(tree, count):
        for c in tree.children:
            if hasattr(c, 'children') and (not c.is_preterminal()):
                if c.label != root and tree.label != root:
                    count[u'{}->{}'.format(tree.label, c.label)] += 1
                _collect(c, count)
            # do not count preterminal
            #else:
            #    count[u'{}->{}'.format(tree.label, c)] += 1
    count = defaultdict(int)
    for tree in tree_list:
        _collect(tree, count)
    return count

def step(tree_list, replaced_label, root):
    count = collect_structs(tree_list, root)
    max_struct = max(count, key=count.get)
    print max_struct
    replace(tree_list, max_struct, replaced_label)

def BPE(tree_list, max_iter=10, root='ROOT'):
    label = 1
    for i in range(max_iter):
        step(tree_list, "%d" % label, root)
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
        tree_list.append(t.root)
    #for t in tree_list:
    #    print t.to_parse_string()

    BPE(tree_list, max_iter=5, root='ROOT')

    for t in tree_list:
        print t.to_parse_string()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_file", required=True, type=str)
    parser.add_argument("--piece_file", type=str)
    parser.add_argument("--replace_file", type=str)
    parser.add_argument("--root", type=str, default='ROOT')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--out_file", type=str)

    args = parser.parse_args()

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
        tree_list.append(t.root)
    #for t in tree_list:
    #    print t.to_parse_string()
    if args.replace_file:
        label = 1
        with codecs.open(args.replace_file, encoding='utf-8') as myfile:
            for line in myfile:
                replace(tree_list, line.strip(), "%d" % label)
                label += 1
    else:
        BPE(tree_list, max_iter=args.max_iter, root=args.root)

    with codecs.open(args.out_file, encoding='utf-8', mode='w') as out:
        for t in tree_list:
            out.write(u'{}\n'.format(t.to_parse_string()))