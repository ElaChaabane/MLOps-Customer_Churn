import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

def main(args):
    if args.prepare:
        print("\n🔄 Préparation des données...")
        prepare_data("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv", "~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")
        print("✅ Données préparées et enregistrées !")

    elif args.train:
        print("\n🚀 Chargement et préparation des données...")
        X_train, y_train, X_test, y_test = prepare_data("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv", "~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")

        print("\n🎯 Entraînement du modèle...")
        model = train_model(X_train, y_train)

        print("\n💾 Sauvegarde du modèle...")
        save_model(model)

    elif args.evaluate:
        print("\n📂 Chargement du modèle...")
        model = load_model()

        print("\n📊 Chargement et préparation des données de test...")
        X_train, y_train, X_test, y_test = prepare_data("~/ela_chaabane_4ds2_ml_project/churn-bigml-80.csv", "~/ela_chaabane_4ds2_ml_project/churn-bigml-20.csv")

        print("\n🔍 Évaluation du modèle...")
        evaluate_model(model, X_test, y_test)

# Corrected if condition
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de prédiction du Churn")

    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")

    args = parser.parse_args()
    main(args)

