from xnmt.serializer import Serializable

class BilingualTrainingCorpus(Serializable):
  """
  A structure containing training and development sets for bilingual training
  """

  yaml_tag = "!BilingualTrainingCorpus"
  def __init__(self, train_src, train_trg, dev_src, dev_trg, train_id_file=None, dev_id_file=None):
    self.train_src = train_src.split("|")
    self.train_trg = train_trg.split("|")
    self.train_id_file = train_id_file
    self.dev_src = dev_src.split("|")
    self.dev_trg = dev_trg.split("|")
    print self.train_src
    print self.train_trg
    self.dev_id_file = dev_id_file
    if len(self.train_src) > 1:
      assert len(self.train_src) == len (self.dev_src), "BilingualTrainingCorpus: train/dev src length does not match"
    else:
      self.train_src = self.train_src[0]
      self.dev_src = self.dev_src[0]
    if len(self.train_trg) > 1:
      assert len(self.train_trg) == len (self.dev_trg), "BilingualTrainingCorpus: train/dev trg length does not match"
    else:
      self.train_trg = self.train_trg[0]
      self.dev_trg = self.dev_trg[0]
