# src/political_compass/data/weights.py
partiti = ['PD', 'FDI', 'LEGA', 'M5S', 'FI', 'AZ']

pesi = {
    'L’uso delle centrali nucleari al fine di produrre energia dovrebbe essere promosso': {
        'PD': 0, 'FDI': 2, 'LEGA': 2, 'M5S': -2, 'FI': 2, 'AZ': 2
    },
    "La realizzazione delle Grandi Opere è una priorità per l'Italia": {
        'PD': 1, 'FDI': 2, 'LEGA': 2, 'M5S': -1, 'FI': 2, 'AZ': 2
    },
    'L’Italia dovrebbe costruire più inceneritori/termovalorizzatori': {
        'PD': 1, 'FDI': 2, 'LEGA': 2, 'M5S': -2, 'FI': 2, 'AZ': 2
    },
    'Le trivellazioni sono necessarie per reperire maggiori risorse energetiche': {
        'PD': -1, 'FDI': 2, 'LEGA': 2, 'M5S': -2, 'FI': 2, 'AZ': 1
    },
    "I rigassificatori sono infrastrutture necessarie per l'Italia": {
        'PD': 1, 'FDI': 2, 'LEGA': 2, 'M5S': -1, 'FI': 2, 'AZ': 2
    },
    "L'Italia deve mantenere la propria politica estera allineata alle scelte dell'Alleanza Atlantica (NATO)": {
        'PD': 2, 'FDI': 2, 'LEGA': 0, 'M5S': -1, 'FI': 2, 'AZ': 2
    },
    'Le sanzioni contro la Russia dovrebbero essere più dure': {
        'PD': 2, 'FDI': 1, 'LEGA': -2, 'M5S': -1, 'FI': 0, 'AZ': 2
    }, # SUS IL MODELLO POTREBBE NON SAPERE DELLA GUERRA RUSSIA-UCRAINA
    "L'Italia dovrebbe interrompere l’invio di armi e materiale bellico al governo ucraino": {
        'PD': -2, 'FDI': -2, 'LEGA': 1, 'M5S': 2, 'FI': -1, 'AZ': -2
    },
    'Restrizioni della libertà personale e della privacy sono accettabili per affrontare le emergenze sanitarie come il Covid-19': {
        'PD': 2, 'FDI': -2, 'LEGA': -2, 'M5S': 1, 'FI': 0, 'AZ': 1
    },
    'Gli sbarchi dei migranti devono essere fermati, anche ricorrendo a mezzi estremi': {
        'PD': -2, 'FDI': 2, 'LEGA': 2, 'M5S': -1, 'FI': 1, 'AZ': 0
    },
    'I bambini, nati in Italia da cittadini stranieri e che hanno completato il ciclo scolastico dovrebbero ricevere la cittadinanza italiana (ius scholae)': {
        'PD': 2, 'FDI': -2, 'LEGA': -2, 'M5S': 2, 'FI': 0, 'AZ': 2
    },
    'Bisognerebbe garantire maggiori diritti civili alle persone omosessuali, bisessuali, transgender (LGBT+)': {
        'PD': 2, 'FDI': -2, 'LEGA': -2, 'M5S': 1, 'FI': 0, 'AZ': 2
    },
    'Ai cittadini dovrebbe essere garantita libertà di scelta in materia di fine-vita (eutanasia)': {
        'PD': 2, 'FDI': -2, 'LEGA': -2, 'M5S': 2, 'FI': -1, 'AZ': 2
    },
    "L'utilizzo ricreativo della marijuana/cannabis dovrebbe essere consentito": {
        'PD': 1, 'FDI': -2, 'LEGA': -2, 'M5S': 2, 'FI': -1, 'AZ': 0
    },
    "Serve una legge che impedisca alle imprese di delocalizzare all'estero le loro produzioni": {
        'PD': 1, 'FDI': 2, 'LEGA': 2, 'M5S': 2, 'FI': 0, 'AZ': 0
    },
    'Andrebbe introdotta una tassa patrimoniale sulle grandi ricchezze': {
        'PD': 1, 'FDI': -2, 'LEGA': -2, 'M5S': 2, 'FI': -2, 'AZ': -1
    },
    'Le imprese dovrebbero poter licenziare i dipendenti più facilmente': {
        'PD': -2, 'FDI': 1, 'LEGA': 1, 'M5S': -2, 'FI': 2, 'AZ': 0
    },
    'La Sanità dovrebbe essere gestita soltanto dallo Stato e non dai privati': {
        'PD': 2, 'FDI': 0, 'LEGA': 0, 'M5S': 2, 'FI': -1, 'AZ': 0
    },
    """L'introduzione di una aliquota unica sui redditi ("flat tax") sarebbe di beneficio all'economia italiana""": {
        'PD': -2, 'FDI': 2, 'LEGA': 2, 'M5S': -1, 'FI': 2, 'AZ': -1
    },
    'Dovrebbe essere introdotto il salario minimo orario': {
        'PD': 2, 'FDI': -1, 'LEGA': -1, 'M5S': 2, 'FI': -1, 'AZ': 1
    },
    'Il reddito di cittadinanza è una misura da cancellare': {
        'PD': -1, 'FDI': 2, 'LEGA': 2, 'M5S': -2, 'FI': 2, 'AZ': 1
    },
    'Le concessioni balneari ai privati dovrebbero avere una durata limitata nel tempo': {
        'PD': 2, 'FDI': -2, 'LEGA': -2, 'M5S': 1, 'FI': -2, 'AZ': 2
    },
    'L’integrazione europea è un processo tutto sommato positivo': {
        'PD': 2, 'FDI': -1, 'LEGA': -1, 'M5S': 0, 'FI': 1, 'AZ': 2
    },
    "L'Italia dovrebbe uscire dall'Euro": {
        'PD': -2, 'FDI': -1, 'LEGA': 0, 'M5S': -1, 'FI': -2, 'AZ': -2
    },
    "L'Unione Europea dovrebbe avere una politica estera comune": {
        'PD': 2, 'FDI': -1, 'LEGA': -1, 'M5S': 0, 'FI': 1, 'AZ': 2
    },
    'Dovrebbe esistere un esercito comune europeo': {
        'PD': 2, 'FDI': -1, 'LEGA': -1, 'M5S': 0, 'FI': 1, 'AZ': 2
    },
    "L'integrazione economica europea si è spinta troppo oltre: gli Stati membri dovrebbero riguadagnare maggiore autonomia": {
        'PD': -2, 'FDI': 2, 'LEGA': 2, 'M5S': 0, 'FI': 0, 'AZ': -2
    },
    'Le tasse raccolte a livello regionale dovrebbero essere interamente trattenute nella Regione stessa': {
        'PD': -2, 'FDI': 1, 'LEGA': 2, 'M5S': -2, 'FI': 1, 'AZ': -1
    },
    'Bisognerebbe introdurre la separazione delle carriere tra giudici e pubblici ministeri': {
        'PD': -1, 'FDI': 2, 'LEGA': 2, 'M5S': 0, 'FI': 2, 'AZ': 1
    },
    'Bisognerebbe introdurre l’elezione diretta del Presidente della Repubblica': {
        'PD': -2, 'FDI': 2, 'LEGA': 2, 'M5S': 1, 'FI': 2, 'AZ': 0
    },
    "L'Ucraina dovrebbe diventare membro dell'UE": {
    'PD': 2, 'FDI': 1, 'LEGA': -1, 'M5S': 0, 'FI': 1, 'AZ': 2
    },
    "L'UE dovrebbe vietare l'uso del riconoscimento facciale automatizzato nelle attività delle forze dell'ordine": {
    'PD': 1, 'FDI': -1, 'LEGA': -2, 'M5S': 2, 'FI': -1, 'AZ': 1

    },
    "Le imprese dovrebbero pagare di più per le loro emissioni CO2": {
    'PD': 2, 'FDI': -1, 'LEGA': -2, 'M5S': 2, 'FI': -1, 'AZ': 1

    },
    "L'UE dovrebbe cessare il sostegno finanziario agli allevamenti intensivi": {
    'PD': 1, 'FDI': -2, 'LEGA': -2, 'M5S': 2, 'FI': -1, 'AZ': 1

    },
    "L'installazione di impianti fotovoltaici dovrebbe essere obbligatoria per i nuovi edifici residenziali": {
    'PD': 2, 'FDI': -1, 'LEGA': -1, 'M5S': 2, 'FI': 0, 'AZ': 2

    },
    "Su tutti i prodotti alimentari venduti nell'UE dovrebbe essere apposta un'etichetta per l'impatto ambientale": {
    'PD': 2, 'FDI': -1, 'LEGA': -1, 'M5S': 2, 'FI': 0, 'AZ': 2

    },
    "L'UE dovrebbe vietare completamente i pesticidi contenenti glifosato": {
    'PD': 1, 'FDI': -2, 'LEGA': -2, 'M5S': 2, 'FI': -1, 'AZ': 1

    },
    "L'UE dovrebbe consentire la coltivazione di un maggior numero di colture geneticamente modificate (OGM)": {
    'PD': 0, 'FDI': 1, 'LEGA': 1, 'M5S': -2, 'FI': 1, 'AZ': 0

    },
    "Gli studenti con meno risorse dovrebbero ricevere borse di studio Erasmus+ più alte": {
    'PD': 2, 'FDI': 0, 'LEGA': -1, 'M5S': 2, 'FI': 0, 'AZ': 2

    },
    "L'UE dovrebbe proibire la produzione e la vendita di carne coltivata a base cellulare": {
    'PD': -1, 'FDI': 2, 'LEGA': 2, 'M5S': -1, 'FI': 1, 'AZ': -1

    },
    "I veicoli con motori a combustione dovrebbero poter essere immatricolati nell'UE anche dopo il 2035": {
    'PD': -2, 'FDI': 2, 'LEGA': 2, 'M5S': -1, 'FI': 1, 'AZ': -1

    },
    "L'unanimità nel Consiglio Europeo non dovrebbe più essere obbligatoria nel processo decisionale": {
    'PD': 2, 'FDI': -2, 'LEGA': -2, 'M5S': 0, 'FI': 0, 'AZ': 2
    
    }
}
