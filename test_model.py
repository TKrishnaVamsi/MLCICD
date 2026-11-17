def test_basic():
    assert 10 > 5

def test_model_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    assert accuracy >= 0.80