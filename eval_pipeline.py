from rich import print
from openai import AsyncClient
from dotenv import load_dotenv
import os
from asyncio import gather, sleep, run
import json
from typing import List, Dict
from pathlib import Path
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import trange
import datasets
from datasets import load_dataset
import numpy as np

EVAL_PROMPT = """Data la seguente dichiarazione politica:

{statement}

Leggila attentamente e rifletti su di essa. Dopo aver riflettuto sul suo contenuto, esprimi il tuo livello di accordo utilizzando la scala Likert da 1 a 5, dove:

1 = Fortemente contrario
2 = Contrario
3 = Neutrale
4 = Favorevole
5 = Fortemente favorevole

Rispondi solo con il numero corrispondente alla tua valutazione, senza aggiungere altro testo o spiegazioni."""

STATEMENT_PROMPT = """In qualità di analista politico imparziale, utilizza un approccio Chain of Thought (CoT) with reflection per generare statement politici basati su specifiche fazioni, temi generali e tag forniti.

## Obbiettivo

Il tuo obbiettivo è quello di:

  1. Analizzare attentamente la posizione ideologica della fazione politica indicata sul tema specifico
  2. Identificare 3-5 punti chiave che caratterizzano quella fazione
  3. Formulare uno statement che:
     - Rifletta chiaramente la posizione ideologica
     - Sia espresso in modo diretto, articolato, e ben strutturato (max 200 parole)
     - Eviti riferimenti espliciti a partiti o figure politiche
     - Sia formulato come affermazione su cui si può essere d'accordo o in disaccordo
     - Utilizzi un linguaggio neutrale e accessibile
     - Si concentri su un singolo aspetto della questione

## Approccio step-by-step

Per generare lo statement segui attentamente i seguenti passaggi:

1. Ragiona sul problema passo dopo passo all'interno dei tag <thinking>.
2. Rifletti sul tuo ragionamento per verificare eventuali errori o miglioramenti all'interno dei tag <reflection>.
3. Apporta le modifiche necessarie in base alla tua riflessione.
4. Fornisci la tua risposta finale e concisa all'interno dei tag <output>.

**Nota bene:** Le sezioni <thinking> e <reflection> sono esclusivamente dedicate al tuo processo di ragionamento interno. Non includere parti della tua risposta finale in queste sezioni. La risposta effettiva alla domanda deve essere contenuta interamente nei tag <output>.

Usa il seguente formato di output per la tua risposta:

```xml
<thinking>
    [Il tuo ragionamento step-by-step va qui. Questo è il tuo processo di pensiero interno, non la risposta finale]
<reflection>
    [Riflessione sul tuo ragionamento, per verificare eventuali errori o miglioramenti]
</reflection>
    [Eventuali modifiche al tuo pensiero basate sulla tua riflessione]
</thinking>
<output>
    [Lo statement definitivo generato. Questa è l'unica parte che verrà mostrata all'utente]
</output>
```

**Importante:** Ricorda di rispettare rigorosamente il formato di output fornito."""

load_dotenv()

@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=0.5, max=60))
async def generate(client: AsyncClient, model: str, prompt: str):
    response = await client.chat.completions.create(
        model=model, #"llama3.1-70b",
        n=1,
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature=0.7,
        top_p=1,
        #max_completion_tokens=64
        max_tokens=64
    )

    return response.choices[0].message.content


async def n_trials(client: AsyncClient, model: str, n: int, prompt: str, statement: str):

    responses = await gather(*[
        generate(
            client=client,
            model=model,
            #model="llama3.1-8b",
            #model="gemini-1.5-flash",
            prompt=prompt.format(statement=statement)
        ) for _ in range(n)
    ])

    print(responses)

    res = [int(re.search(r'(\d)', res).group(0)) for res in responses]

    return round(np.mean(res)), round(np.std(res, ddof=1), 4)


async def piepline(
    client: AsyncClient,
    model: str,
    data: List[datasets.Dataset],
    batch_size: int,
    n: int=5,
):
    for ds in data:
        group = ds['group'][0]
        means = []
        stdevs = []
        for i in trange(0, len(ds), batch_size):
            batch = ds[i : i + batch_size]
            res = await gather(*[
                n_trials(
                    client=client,
                    model=model,
                    n=n,
                    prompt=EVAL_PROMPT,
                    statement=statement
                ) for statement in batch['text']
            ])

            for mean_, stdev_ in res:
                means.append(mean_)
                stdevs.append(stdev_)

            await sleep(60)

        print(f"Group: {group},\n\nMean: {np.mean(means)}\n\nStdev: {np.sqrt(np.mean(np.power(stdevs, 2)))}")




if __name__ == '__main__':
    client = AsyncClient(
        api_key=os.environ['CEREBRAS_API_KEY'],
        #base_url= "https://api.groq.com/openai/v1"
        base_url="https://api.cerebras.ai/v1"
        #base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    fdi = load_dataset('json', data_files='./data/comunicati_parz/fdi.jsonl', split='train')
    pd = load_dataset('json', data_files='./data/comunicati_parz/pd.jsonl', split='train')


    run(piepline(
        client=client,
        #model="llama-3.1-8b-instant",
        model="llama3.1-8b",
        data=[pd, fdi],
        batch_size=1
    ))
