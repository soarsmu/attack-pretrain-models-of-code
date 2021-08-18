import argparse
from parser_folder.DFG import DFG_python, DFG_java, DFG_c
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)
from tree_sitter import Language, Parser
import sys
sys.path.append('.')
sys.path.append('../')
from utils import is_valid_variable_name

path = 'parser_folder/my-languages.so'
c_code = """
static void do_busid_cmd(ESPState *s, uint8_t *buf, uint8_t busid)
{
    int32_t datalen;
    int lun;
    DPRINTF("do_busid_cmd: busid 0x%x\n", busid);
    lun = busid & 7;
    s->current_req = scsi_req_new(s->current_dev, 0, lun, NULL);
    datalen = scsi_req_enqueue(s->current_req, buf);
    s->ti_size = datalen;
    if (datalen != 0)
    {
        s->rregs[ESP_RSTAT] = STAT_TC;
        s->dma_left = 0;
        s->dma_counter = 0;
        if (datalen > 0)
        {
            s->rregs[ESP_RSTAT] |= STAT_DI;
        }
        else
        {
            s->rregs[ESP_RSTAT] |= STAT_DO;
        }
        scsi_req_continue(s->current_req);
    }
    s->rregs[ESP_RINTR] = INTR_BS | INTR_FC;
    s->rregs[ESP_RSEQ] = SEQ_CD;
    esp_raise_irq(s);
}"""

python_code = """ 
a = "abcdefghijklmnopqrstuvwxyz" 
d = { } 
for c in a : 
d [ c ] = "*" 
i = "ejp mysljylc kd kxveddknmc re jsicpdrysi" 
o = "our language is impossible to understand" 
for k , v in zip ( i , o ) : 
d [ k ] = v 
i = "rbcpc ypc rtcsra dkh wyfrepkym veddknkmkrkcd" 
o = "there are twenty six factorial possibilities" 
for k , v in zip ( i , o ) : 
d [ k ] = v 
i = "de kr kd eoya kw aej tysr re ujdr lkgc jv" 
o = "so it is okay if you want to just give up" 
for k , v in zip ( i , o ) : 
d [ k ] = v 
i = "y qee" 
o = "a zoo" 
for k , v in zip ( i , o ) : 
d [ k ] = v 
d [ 'z' ] = 'q' 
import sys 
f = file ( "A-small-attempt1.in" ) 
w = file ( "answer.txt" , "w" ) 
cnt = int ( f . readline ( ) [ : - 1 ] ) 
for no in range ( cnt ) : 
i = f . readline ( ) [ : - 1 ] 
o = "" 

for k in i :
o += d [ k ] 
print >> w , "Case #%d:" % ( no + 1 ) , o

"""

java_code = """
package org . genie . java_awt ; public class HierarchyBoundsAdapterProxy extends java . awt . event . HierarchyBoundsAdapter { long swiftObject ; HierarchyBoundsAdapterProxy ( long swiftObject ) { super ( ) ; this . swiftObject = swiftObject ; } public native void __ancestorMoved ( java . awt . event . HierarchyEvent e ) ; boolean entered_ancestorMoved_0 ; public void ancestorMoved ( java . awt . event . HierarchyEvent e ) { if ( ! entered_ancestorMoved_0 ) { entered_ancestorMoved_0 = true ; __ancestorMoved ( e ) ; entered_ancestorMoved_0 = false ; } else super . ancestorMoved ( e ) ; } public native void __ancestorResized ( java . awt . event . HierarchyEvent e ) ; boolean entered_ancestorResized_1 ; public void ancestorResized ( java . awt . event . HierarchyEvent e ) { if ( ! entered_ancestorResized_1 ) { entered_ancestorResized_1 = true ; __ancestorResized ( e ) ; entered_ancestorResized_1 = false ; } else super . ancestorResized ( e ) ; } public native boolean __equals ( java . lang . Object arg0 ) ; boolean entered_equals_2 ; public boolean equals ( java . lang . Object arg0 ) { if ( ! entered_equals_2 ) { entered_equals_2 = true ; boolean __return = __equals ( arg0 ) ; entered_equals_2 = false ; return __return ; } else return super . equals ( arg0 ) ; } public native java . lang . String __toString ( ) ; boolean entered_toString_3 ; public java . lang . String toString ( ) { if ( ! entered_toString_3 ) { entered_toString_3 = true ; java . lang . String __return = __toString ( ) ; entered_toString_3 = false ; return __return ; } else return super . toString ( ) ; } public native int __hashCode ( ) ; boolean entered_hashCode_4 ; public int hashCode ( ) { if ( ! entered_hashCode_4 ) { entered_hashCode_4 = true ; int __return = __hashCode ( ) ; entered_hashCode_4 = false ; return __return ; } else return super . hashCode ( ) ; } }
"""
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language(path, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

codes = {}
codes = {
    'python': python_code,
    'java': java_code,
    'c': c_code,
}

def extract_dataflow(code, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index
    try:
        # call dfg_python here
        DFG, _ = parser[1](root_node, index_to_code, {})
    except:
        DFG = []
    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def parse_string(input):
    if (input.startswith("\"\"\"") and input.endswith("\"\"\"")) or \
            (input.startswith("\'\'\'") and input.endswith("\'\'\'")) or \
            (input.startswith("\'") and input.endswith("\'")) or \
            (input.startswith("\"") and input.endswith("\"")):
        return True
    return False


def get_identifiers(code, lang):

    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    print("dfg")
    for i in dfg:
        print(i)
    ret = []
    ret_set = set()
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret_set.add(d[0])
    for item in ret_set:
        ret.append([item])
    return ret, code_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    parser = parsers[args.lang]
    code = codes[args.lang]
    # extract data flow
    data, _ = get_identifiers(code, args.lang)
    print("final ret")
    for identifier in data:
        print(identifier)


if __name__ == '__main__':
    main()

