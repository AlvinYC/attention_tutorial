from torchtext.data import Dataset, Field, Example, BucketIterator
import subprocess
import codecs
import re
import random

random.seed(30)

def get_xmllcsts(filename,limit=None):
    # regular_expression is suitable for LCSTS PART_I.txt, PART_II.txt, PART_III.txt
    pattern = re.compile(r'''<doc id=(?:\d+)>(?:\n\s+<human_label>(?:\d+)</human_label>)?
    <summary>\n\s+(.+)\n\s+</summary>
    <short_text>\n\s+(.+)\n\s+</short_text>\n</doc>''', re.M)
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = ''.join(f.readlines())
    lcsts_list = re.findall(pattern, content)[:limit]

    return lcsts_list

# pre-processing, Ian's Global-Encoding request, reverse src
def reverse_field(this_example):
    return list(reversed(this_example))


def load_dataset_lcsts(batch_size,macbook=False,filename=None):
    filename = './lcsts_xml/PART_I_10000.txt' if filename == None else filename
    TRG = Field(tokenize=list, init_token='<s>', eos_token='</s>', pad_token='<blank>',include_lengths=False)
    SRC = Field(tokenize=list, pad_token='<blank>', include_lengths=False, preprocessing=reverse_field)
    TRG_LEN = Field(sequential=False,use_vocab=False)
    SRC_LEN = Field(sequential=False,use_vocab=False)
    TRG_ORI = Field(sequential=False)
    SRC_ORI = Field(sequential=False)

    fields = [('trg',TRG),('src',SRC),('trg_len',TRG_LEN),('src_len',SRC_LEN),('trg_ori',TRG_ORI),('src_ori',SRC_ORI)] 
    lcsts_list = get_xmllcsts(filename)

    examples = list(map(lambda x :Example.fromlist([x[0],x[1],str(len(x[0])),str(len(x[1])),x[0],x[1]],fields),lcsts_list))
    all_data = Dataset(examples=examples,fields=fields) 
    train, val, test = all_data.split(split_ratio=[0.8,0.1,0.1],random_state=random.getstate())   
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    # src do not contain <sos> <eos> , need manual build specials
    # according to Ian's Global-Encoding code, we have the following defintion
    # PAD = 0 PAD_WORD = '<blank>'
    # UNK = 1 UNK_WORD = '<unk> '
    # BOS = 2 BOS_WORD = '<s>'
    # EOS = 3 EOS_WORD = '</s>'    
    # since torchtext use <unk> as vocabulary index 0, <blank> as 1
    # we will get different pad value (numerical), Ian's Global-Encodng pad = 0, torchtext pad = 1

    SRC.build_vocab(train.src, min_freq=2, specials=['<blank>','<unk>','<s>','</s>'])
    TRG.build_vocab(train.trg, max_size=10000, specials=['<blank>','<unk>','<s>','</s>'])
    TRG_ORI.build_vocab(all_data.trg_ori)
    SRC_ORI.build_vocab(all_data.src_ori)

    LCSTS_FIELD = {'src':SRC, 'trg':TRG, 'src_ori':SRC_ORI, 'trg_ori':TRG_ORI}
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_sizes=(batch_size,batch_size,batch_size), repeat=False, shuffle=False, sort_key=lambda x: len(x.src))
    
    return train_iter, val_iter, test_iter, LCSTS_FIELD

if __name__ == "__main__":
    lcsts_path = '/Users/alvin/workspace/dataset_nlp/lcsts/LCSTS_DATA_XML/PART_I_10000.txt'
    train_iter, val_iter, test_iter, LCSTS_FIELD = load_dataset_lcsts(batch_size=5,filename=lcsts_path)
    tl_batch = next(iter(train_iter))
    print('complete')