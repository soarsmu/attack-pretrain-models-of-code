package JavaExtractor.Visitors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@SuppressWarnings("StringEquality")
public class FunctionVisitor extends VoidVisitorAdapter<Object> {
	private ArrayList<MethodContent> m_Methods = new ArrayList<>();
        

	@Override
	public void visit(MethodDeclaration node, Object arg) {
                visitMethod(node, arg);

		super.visit(node, arg);
	}

	private void visitMethod(MethodDeclaration node, Object obj) {
		LeavesCollectorVisitor leavesCollectorVisitor = new LeavesCollectorVisitor();
		leavesCollectorVisitor.visitDepthFirst(node);
		ArrayList<Node> leaves = leavesCollectorVisitor.getLeaves();

		String normalizedMethodName = Common.normalizeName(node.getName(), Common.BlankWord);
		ArrayList<String> splitNameParts = Common.splitToSubtokens(node.getName());
		String splitName = normalizedMethodName;
		if (splitNameParts.size() > 0) {
			splitName = splitNameParts.stream().collect(Collectors.joining(Common.internalSeparator));
		}

		if (node.getBody() != null) {
                    MethodContent methodContent = new MethodContent(leaves, splitName, getMethodLength(node.getBody().toString()));
                    
                    Set<String> variables = new HashSet<>();
                    
                    node.accept(new VoidVisitorAdapter<Void>(){
                        @Override
                        public void visit(VariableDeclarationExpr n, Void arg) {
                            Set<String> myVars = n.getVariables().stream()
									.map(var -> Common.normalizeName(var.getId().getName(), Common.BlankWord))
									.collect(Collectors.toSet());
                            variables.addAll(myVars);
                            super.visit(n, arg); //To change body of generated methods, choose Tools | Templates.
                        }
                        @Override
                        public void visit(Parameter n, Void arg) {
                            variables.add(Common.normalizeName(n.getName(), Common.BlankWord));
                            super.visit(n, arg); //To change body of generated methods, choose Tools | Templates.
                        }
                        
                    } ,null);                    
                    
                    methodContent.setVarNames(variables);
                    m_Methods.add(methodContent);
		}
	}

	private long getMethodLength(String code) {
		String cleanCode = code.replaceAll("\r\n", "\n").replaceAll("\t", " ");
		if (cleanCode.startsWith("{\n"))
			cleanCode = cleanCode.substring(3).trim();
		if (cleanCode.endsWith("\n}"))
			cleanCode = cleanCode.substring(0, cleanCode.length() - 2).trim();
		if (cleanCode.length() == 0) {
			return 0;
		}
		long codeLength = Arrays.asList(cleanCode.split("\n")).stream()
				.filter(line -> (line.trim() != "{" && line.trim() != "}" && line.trim() != ""))
				.filter(line -> !line.trim().startsWith("/") && !line.trim().startsWith("*")).count();
		return codeLength;
	}

	public ArrayList<MethodContent> getMethodContents() {
		return m_Methods;
	}
}
