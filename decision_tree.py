import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Fetch the breast cancer dataset
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

proportions = [0.6, 0.4, 0.2, 0.1]
for proportion in proportions:
	# Shuffle and split the data into training and testing set
	feature_train, feature_test, label_train, label_test = train_test_split(
		X, y, test_size=proportion, stratify=y, random_state=42
	)

	clf = DecisionTreeClassifier(criterion='entropy')
	clf.fit(feature_train, label_train)
	# print(clf)

	# Export the decision tree to a DOT file
	dot_data = export_graphviz(clf, out_file=None,
							   feature_names=breast_cancer.data.attribute_names,
							   class_names=breast_cancer.data.class_names,
							   filled=True, rounded=True, special_characters=True)

	# Save the DOT file for later visualization
	# with open("breast_cancer_tree.dot", "w") as f:
	#     f.write(dot_data)

	graph = graphviz.Source(dot_data)
	graph.render(f"decision_tree_test_size_{proportion}", format="pdf", cleanup=True)  # Replace with desired filename (without extension)
