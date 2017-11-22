from vocab import *
from input import *
import sys



if __name__ == "__main__":
  parse = sys.argv[1]
  filename = parse
  if len(sys.argv) > 2:
    filename += ("," + sys.argv[2])

  tree_reader = TreeReader();
  count, data_len = 0, 0
  max_data_len, min_data_len = -1, -1
  max_data = []
  max_idx = -1
  for tree_input in tree_reader.read_sents(filename):
    count += 1
    data_len += len(tree_input)
    if max_data_len < 0 or len(tree_input) > max_data_len:
      max_data_len = len(tree_input)
      max_idx = count
    if min_data_len < 0 or len(tree_input) < min_data_len:
      min_data_len = len(tree_input)
  print "max idx: ", max_idx
  print "ave data len: ", (data_len + 0.0) / count
  print "max data len: ", max_data_len
  print "min data len: ", min_data_len

