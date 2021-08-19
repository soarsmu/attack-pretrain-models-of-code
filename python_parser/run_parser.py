import argparse
import csv
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
	static int img_read_packet(AVFormatContext *s1, AVPacket *pkt) { VideoData *s = s1->priv_data; char filename[1024]; int ret; ByteIOContext f1, *f; 
}

"""
python_code = """ 
static void RENAME(vertical_compose53iL0)(uint8_t *_b0, uint8_t *_b1, uint8_t *_b2,\n\n                                          int width)\n\n{\n\n    int i;\n\n    TYPE *b0 = (TYPE *)_b0;\n\n    TYPE *b1 = (TYPE *)_b1;\n\n    TYPE *b2 = (TYPE *)_b2;\n\n    for (i = 0; i < width; i++)\n\n        b1[i] -= (b0[i] + b2[i] + 2) >> 2;\n\n
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
    code = codes[args.lang]
    data, _ = get_identifiers(code, args.lang)
    print("final ret")
    for identifier in data:
        print(identifier)


if __name__ == '__main__':
    main()

