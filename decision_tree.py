import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# # Breast Cancer Dataset
# breast_cancer_wisconsin_diagnostic = load_breast_cancer()
# bc_features = pd.DataFrame(breast_cancer_wisconsin_diagnostic.data, columns=breast_cancer_wisconsin_diagnostic.feature_names)  # Các đặc trưng
# bc_labels = pd.Series(breast_cancer_wisconsin_diagnostic.target, name="label")  # Nhãn: 0 = Malignant, 1 = Benign
#
# # White Wine Quality Dataset
# file_path = './data/wine+quality/winequality-white.csv'
# w_wine_data = pd.read_csv(file_path, sep=';')
# w_wine_features = w_wine_data.iloc[:, :-1]
# w_wine_labels = w_wine_data['quality']
#
# # Red Wine Quality Dataset
# file_path = './data/wine+quality/winequality-red.csv'
# r_wine_data = pd.read_csv(file_path, sep=';')
# r_wine_features = r_wine_data.iloc[:, :-1]
# r_wine_labels = r_wine_data['quality']

# Titanic Dataset
file_path = './data/Titanic/Titanic-Dataset.csv'
titanic_data = pd.read_csv(file_path)
titanic_features = titanic_data.iloc[:, :-1]
titanic_labels = titanic_data['Survived']


def prepare_datasets(features, labels):
	subsets = {}  # Sử dụng dictionary thay vì danh sách để dễ dàng lưu trữ các tập
	# Chia dữ liệu với các tỷ lệ khác nhau và lưu vào dictionary
	feature_train_40, feature_test_60, label_train_40, label_test_60 = train_test_split(features, labels, train_size=0.4, stratify=labels,
																						shuffle=True, random_state=0)
	subsets['train_40'] = (feature_train_40, label_train_40)
	subsets['test_60'] = (feature_test_60, label_test_60)

	feature_train_60, feature_test_40, label_train_60, label_test_40 = train_test_split(features, labels, train_size=0.6, stratify=labels,
																						shuffle=True, random_state=0)
	subsets['train_60'] = (feature_train_60, label_train_60)
	subsets['test_40'] = (feature_test_40, label_test_40)

	feature_train_80, feature_test_20, label_train_80, label_test_20 = train_test_split(features, labels, train_size=0.8, stratify=labels,
																						shuffle=True, random_state=0)
	subsets['train_80'] = (feature_train_80, label_train_80)
	subsets['test_20'] = (feature_test_20, label_test_20)

	feature_train_90, feature_test_10, label_train_90, label_test_10 = train_test_split(features, labels, train_size=0.9, stratify=labels,
																						shuffle=True, random_state=0)
	subsets['train_90'] = (feature_train_90, label_train_90)
	subsets['test_10'] = (feature_test_10, label_test_10)

	return subsets


# # Tạo các tập huấn luyện và kiểm tra cho Breast Cancer Dataset
# bc_datasets = prepare_datasets(bc_features, bc_labels)
#
# # Tạo các tập huấn luyện và kiểm tra cho White Wine Quality Dataset
# w_wine_datasets = prepare_datasets(w_wine_features, w_wine_labels)
#
# # Tạo các tập huấn luyện và kiểm tra cho Red Wine Quality Dataset
# r_wine_datasets = prepare_datasets(r_wine_features, r_wine_labels)

# Tạo các tập huấn luyện và kiểm tra cho Titanic Dataset
titanic_datasets = prepare_datasets(titanic_features, titanic_labels)


def build_decision_tree(feature_train, label_train):
	clf = DecisionTreeClassifier(criterion='entropy')
	clf.fit(feature_train, label_train)
	return clf


def export_decision_tree_graphviz(dataset, format):
	train = ["train_40", "train_60", "train_80", "train_90"]
	test = ["test_60", "test_40", "test_20", "test_10"]
	for train_key, test_key in zip(train, test):
		decision_tree = build_decision_tree(*dataset[train_key])
		dot_data = export_graphviz(decision_tree, out_file=None,
								   filled=True, rounded=True, special_characters=True)

		graph = graphviz.Source(dot_data)
		graph.render(f"decision_tree_{train_key}_{test_key}", format=format, cleanup=True)  # Replace with desired filename (without extension)


# export_decision_tree_graphviz(bc_datasets, f"decision_tree_bc", "png")
# export_decision_tree_graphviz(w_wine_datasets, f"decision_tree_w_wine", "svg")
export_decision_tree_graphviz(titanic_datasets, "svg")

