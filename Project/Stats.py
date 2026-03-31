def print_statistics(D0, D1):
    classi = [("False Fingerprint", D0), ("True Fingerprint", D1)]
    nomi_misure = ["F1", "F2", "F3", "F4", "F5", "F6"]
    
    for nome_classe, matrice_dati in classi:
        print(f"========================================")
        print(f" CLASSE: {nome_classe}")
        print(f"========================================")
        medie = matrice_dati.mean(1)
        varianze = matrice_dati.var(1)
        dev_standard = matrice_dati.std(1)
        
        for i in range(6):
            print(f"- {nomi_misure[i]}:")
            print(f"    Media: {medie[i]:.4f}")
            print(f"    Varianza: {varianze[i]:.4f}")
            print(f"    Dev. Standard: {dev_standard[i]:.4f}")
        print("\n") 