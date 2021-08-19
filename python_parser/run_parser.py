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

path = '../../../python_parser/parser_folder/my-languages.so'
c_code = """
static void RENAME(vertical_compose53iL0)(uint8_t *_b0, uint8_t *_b1, uint8_t *_b2,\n\n                                          int width)\n\n{\n\n    int i;\n\n    TYPE *b0 = (TYPE *)_b0;\n\n    TYPE *b1 = (TYPE *)_b1;\n\n    TYPE *b2 = (TYPE *)_b2;\n\n    for (i = 0; i < width; i++)\n\n        b1[i] -= (b0[i] + b2[i] + 2) >> 2;\n\n
"""

python_code = """ 
import sys import time import itertools def iterate_cases_1lpc ( filepath ) : with file ( filepath , 'rb' ) as f_in : for line_index , line in enumerate ( f_in ) : if line_index == 0 : continue yield line_index , line . strip ( ) . split ( ' ' ) def iterate_cases_nlpc ( filepath , n ) : with file ( filepath , 'rb' ) as f_in : case_counter = 1 case = [ ] for line_index , line in enumerate ( f_in ) : if line_index == 0 : continue case . append ( line . strip ( ) . split ( ' ' ) ) if not line_index % n : yield case_counter , case case_counter += 1 case = [ ] def iterate_cases_glpc ( filepath ) : with file ( filepath , 'rb' ) as f_in : case_counter = 0 new_case = True for line_index , line in enumerate ( f_in ) : if line_index == 0 : continue if new_case : new_case = False case_counter += 1 case = [ ] assert len ( line . strip ( ) . split ( ' ' ) ) == 1 lines_left = int ( line . strip ( ) ) if not lines_left : new_case = True yield case_counter , case continue if lines_left : lines_left -= 1 case . append ( line . strip ( ) . split ( ' ' ) ) if not lines_left : new_case = True yield case_counter , case def part_of_list_to_int ( array , flags ) : assert len ( array ) == len ( flags ) output = [ ] for index , elem in enumerate ( array ) : if flags [ index ] : output . append ( int ( elem ) ) else : output . append ( elem ) return output def list_to_int ( array ) : return part_of_list_to_int ( array , [ True ] * len ( array ) ) def part_of_list_to_float ( array , flags ) : assert len ( array ) == len ( flags ) output = [ ] for index , elem in enumerate ( array ) : if flags [ index ] : output . append ( float ( elem ) ) else : output . append ( elem ) return output def list_to_float ( array ) : return part_of_list_to_float ( array , [ True ] * len ( array ) ) def get_max_array_on_index ( array , index ) : elem_len = len ( array [ 0 ] ) assert index < elem_len for elem in array : assert elem_len == len ( elem ) max_sub = array [ 0 ] [ index ] max_elem = array [ 0 ] for elem in array : if elem [ index ] > max_sub : max_sub = elem [ index ] max_elem = elem return max_elem def list_index_in_sorted_with_position ( a_list , value , pos ) : list_len = len ( a_list ) if list_len == 1 : if a_list [ 0 ] == value : return pos return - 1 if a_list [ list_len / 2 ] > value : return list_index_in_sorted_with_position ( a_list [ : ( list_len / 2 ) ] , value , pos ) else : return list_index_in_sorted_with_position ( a_list [ ( list_len / 2 ) : ] , value , pos + ( list_len / 2 ) ) def list_index_in_sorted_list ( a_list , value ) : return list_index_in_sorted_with_position ( a_list , value , 0 ) def check_row ( row ) : if row . count ( 'X' ) == 4 : return 'X' if row . count ( 'O' ) == 4 : return 'O' if row . count ( 'X' ) == 3 and row . count ( 'T' ) == 1 : return 'X' if row . count ( 'O' ) == 3 and row . count ( 'T' ) == 1 : return 'O' if row . count ( '.' ) > 0 : return '.' return 'F' def calc_result ( case ) : case = case [ : - 1 ] print "\t%s" % case rows = [ ] for row in case : rows . append ( row [ 0 ] ) for i in xrange ( 4 ) : column = "" for j in xrange ( 4 ) : column += case [ j ] [ 0 ] [ i ] rows . append ( column ) diag1 = "" diag2 = "" for i in xrange ( 4 ) : diag1 += case [ i ] [ 0 ] [ i ] diag2 += case [ 3 - i ] [ 0 ] [ i ] rows . append ( diag1 ) rows . append ( diag2 ) res = [ ] for row in rows : res . append ( check_row ( row ) ) if res . count ( 'X' ) : if res . count ( 'O' ) : raise IOError ( 'both X and O won' ) else : result = "X won" else : if res . count ( 'O' ) : result = "O won" else : if res . count ( '.' ) : result = "Game has not completed" else : result = "Draw" print "\t%s" % res print "\t%s" % result return result def main ( filepath ) : start_time = time . time ( ) with file ( 'output.txt' , 'wb' ) as f_out : for case_index , case in iterate_cases_nlpc ( filepath , 5 ) : print "case #%d: time:%.02f" % ( case_index , time . time ( ) - start_time ) result = calc_result ( case ) f_out . write ( "Case #%d: %s\n" % ( case_index , result ) ) if __name__ == '__main__' : main ( sys . argv [ 1 ] )
"""

java_code = """
package database;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

public class DumpMetaData {
    public DumpMetaData(String connectionUrl) throws SQLException, ClassNotFoundException {
        Class.forName("org.apache.derby.jdbc.EmbeddedDriver");
        Connection conn = DriverManager.getConnection(connectionUrl);
        System.out.println("conn: " + conn);
        Statement s = conn.createStatement();
        DatabaseMetaData md = conn.getMetaData();
        List<String> tableNames = new ArrayList<String>(5);
        ResultSet table_rs = md.getTables(null, null, "%", new String[]{"TABLE"});
        while (table_rs.next()) {
            System.out.println("Table: " + table_rs.getString(3));
            tableNames.add(table_rs.getString(3));
        }
        for (String tableName : tableNames) {
            System.out.println("\n\n** Processing table " + tableName + "\n");
            String query = "SELECT * from " + tableName;
            System.out.println(query);
            ResultSet rs = s.executeQuery(query);
            ResultSetMetaData table_meta = rs.getMetaData();
            int columnCount = table_meta.getColumnCount();
            System.out.println("\nColumn meta data for table:");
            List<String> columnNames = new ArrayList<String>(10);
            columnNames.add("");
            for (int col = 1; col <= columnCount; col++) {
                System.out.println("Column " + col + " name: " + table_meta.getColumnLabel(col));
                System.out.println("  column data type: " + table_meta.getColumnTypeName(col));
                columnNames.add(table_meta.getColumnLabel(col));
            }
            System.out.println("\nFirst row in table:");
            if (rs.next()) {
                for (int col = 1; col <= columnCount; col++) {
                    System.out.println("   " + columnNames.get(col) + ": " + rs.getString(col));
                }
            }
        }
    }
	public void add(){
    }
    public static void main(String[] args) throws SQLException, ClassNotFoundException {
        new DumpMetaData("jdbc:derby:test_database");
    }
}

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

    DFG, _ = parser[1](root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
def parse_string(input):
    if (input.startswith("\"\"\"") and input.endswith("\"\"\"")) or \
            (input.startswith("\'\'\'") and input.endswith("\'\'\'")) or \
            (input.startswith("\'") and input.endswith("\'")) or \
            (input.startswith("\"") and input.endswith("\"")):
        return True
    return False


def get_identifiers(code, lang):

    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    ret_set = set()
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [ [i] for i in ret]
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

