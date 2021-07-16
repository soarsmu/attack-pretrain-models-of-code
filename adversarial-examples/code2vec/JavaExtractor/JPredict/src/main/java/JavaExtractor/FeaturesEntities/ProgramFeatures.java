package JavaExtractor.FeaturesEntities;

import java.util.ArrayList;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonIgnore;
import java.util.HashSet;
import java.util.Set;

public class ProgramFeatures {
	private String name;

	private ArrayList<ProgramRelation> features = new ArrayList<>();
        private Set<String> varNames;

    public Set<String> getVarNames() {
        return varNames;
    }

    public void setVarNames(Set<String> varNames) {
        this.varNames = varNames;
    }

	public ProgramFeatures(String name) {
		this.name = name;
	}

	@SuppressWarnings("StringBufferReplaceableByString")
	@Override
	public String toString() {
		StringBuilder stringBuilder = new StringBuilder();
                stringBuilder.append(varNames.stream().collect(Collectors.joining(","))).append(" ");
		stringBuilder.append(name).append(" ");
		stringBuilder.append(features.stream().map(ProgramRelation::toString).collect(Collectors.joining(" ")));

		return stringBuilder.toString();
	}

	public void addFeature(Property source, String path, Property target) {
		ProgramRelation newRelation = new ProgramRelation(source, target, path);
		features.add(newRelation);
	}

	@JsonIgnore
	public boolean isEmpty() {
		return features.isEmpty();
	}

	public void deleteAllPaths() {
		features.clear();
	}

	public String getName() {
		return name;
	}

	public ArrayList<ProgramRelation> getFeatures() {
		return features;
	}

}
