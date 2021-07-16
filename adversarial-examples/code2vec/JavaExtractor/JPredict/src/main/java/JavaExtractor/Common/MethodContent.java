package JavaExtractor.Common;

import java.util.ArrayList;
import com.github.javaparser.ast.Node;
import java.util.HashSet;
import java.util.Set;

public class MethodContent {
	private ArrayList<Node> leaves;
	private String name;
	private long length;
        private Set<String> varNames = new HashSet<>();

        public Set<String> getVarNames() {
            return varNames;
        }

        public void setVarNames(Set<String> varNames) {
            this.varNames = varNames;
        }


	public MethodContent(ArrayList<Node> leaves, String name, long length) {
		this.leaves = leaves;
		this.name = name;
		this.length = length;
	}

	public ArrayList<Node> getLeaves() {
		return leaves;
	}

	public String getName() {
		return name;
	}

	public long getLength() {
		return length;
	}

}
