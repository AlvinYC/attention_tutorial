from torchtext.data import Dataset, Field, Example, BucketIterator
import subprocess
import codecs
import re

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


def load_dataset_lcsts(batch_size,macbook=False,filename=None):
    filename = './lcsts_xml/PART_I_10000.txt' if filename == None else filename
    TRG = Field(tokenize=list, include_lengths=True, init_token='<sos>', eos_token='<eos>')
    SRC = Field(tokenize=list, include_lengths=True, init_token='<sos>', eos_token='<eos>')
    fields = [('trg',TRG),('src',SRC)] 
    lcsts_list = get_xmllcsts(filename)

    examples = list(map(lambda x :Example.fromlist(x,fields),lcsts_list))
    all_data = Dataset(examples=examples,fields=fields) 
    train, val, test = all_data.split(split_ratio=[0.8,0.1,0.1]) 
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    SRC.build_vocab(train.src, min_freq=2)
    TRG.build_vocab(train.trg, max_size=10000)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False, shuffle=False)
    
    return train_iter, val_iter, test_iter, SRC, TRG

if __name__ == "__main__":
    lcsts_path = '/Users/alvin/workspace/dataset_nlp/lcsts/LCSTS_DATA_XML/PART_I_10000.txt'
    train_iter, val_iter, test_iter, SRC, TRG = load_dataset_lcsts(batch_size=5,filename=lcsts_path)
    tl_batch = next(iter(train_iter))
    print('complete')