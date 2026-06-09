# Confusion analysis — does a wrong (look-alike) compound beat the right one?

Confusion rate = fraction of queries where a hard-negative (chemically-similar wrong compound) ranks above every gold document.

| model | de | en | es | fr | zh | ALL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `alibaba-nlp-gte-multilingual-base` | 61.5% (n=26) | 44.4% (n=27) | 56.0% (n=25) | 57.7% (n=26) | 38.5% (n=26) | 51.5% (n=130) |
| `baai-bge-m3` | 11.1% (n=27) | 7.4% (n=27) | 8.0% (n=25) | 11.1% (n=27) | 26.9% (n=26) | 12.9% (n=132) |
| `cambridgeltl-sapbert-umls-2020ab-all-lang-from-xlmr` | 37.0% (n=27) | 25.9% (n=27) | 28.0% (n=25) | 22.2% (n=27) | 34.6% (n=26) | 29.5% (n=132) |
| `google-embeddinggemma-300m` | 11.1% (n=27) | 7.4% (n=27) | 0.0% (n=25) | 11.1% (n=27) | 3.9% (n=26) | 6.8% (n=132) |
| `ibm-granite-granite-embedding-278m-multilingual` | 11.1% (n=27) | 18.5% (n=27) | 8.0% (n=25) | 22.2% (n=27) | 30.8% (n=26) | 18.2% (n=132) |
| `intfloat-multilingual-e5-large-instruct` | 29.6% (n=27) | 18.5% (n=27) | 40.0% (n=25) | 22.2% (n=27) | 30.8% (n=26) | 28.0% (n=132) |
| `nomic-ai-nomic-embed-text-v2-moe` | 7.4% (n=27) | 14.8% (n=27) | 0.0% (n=25) | 11.1% (n=27) | 19.2% (n=26) | 10.6% (n=132) |
| `qwen-qwen3-embedding-0-6b` | 18.5% (n=27) | 11.1% (n=27) | 4.0% (n=25) | 11.1% (n=27) | 19.2% (n=26) | 12.9% (n=132) |
| `sentence-transformers-labse` | 25.9% (n=27) | 25.9% (n=27) | 20.0% (n=25) | 25.9% (n=27) | 26.9% (n=26) | 25.0% (n=132) |

## Most frequent confusions (winning look-alike, all models)

| right compound | beaten by (look-alike) | relation | count |
| --- | --- | --- | ---: |
| protein polypeptide chain | polypeptide | parent | 28 |
| ethyl | methyl | sibling | 18 |
| propene | ethene | sibling | 18 |
| polysulfur | elemental sulfur | parent | 14 |
| benzene | biphenyl | sibling | 13 |
| sulfide(2-) | sulfate | sibling | 11 |
| methacrylic acid | acrylic acid | sibling | 10 |
| poly(propylene glycol) macromolecule | poly(ethylene glycol) | sibling | 9 |
| poly(vinyl alcohol) macromolecule | poly(alkylene) macromolecule | sibling | 9 |
| adipic acid | succinic acid | sibling | 7 |
| sulfite | sulfate | sibling | 6 |
| poly(vinyl alcohol) macromolecule | polyurethane macromolecule | sibling | 5 |
| sulfide(2-) | sulfite | sibling | 5 |
| benzene | ethanol | sibling | 4 |
| adipic acid | sebacic acid | sibling | 4 |
| poly(propylene glycol) macromolecule | poly(ether) macromolecule | parent | 4 |
| manganese dioxide | metal oxide | parent | 4 |
| carbon dioxide | dioxygen | sibling | 4 |
| sulfonium | ammonium | sibling | 3 |
| methyl | ethyl | sibling | 3 |
| silane | ammonia | sibling | 3 |
| adipic acid | itaconic acid | sibling | 3 |
| polyurethane macromolecule | poly(vinyl alcohol) macromolecule | sibling | 3 |
| phenol | bisphenol | sibling | 3 |
| propene | dioxygen | sibling | 3 |
| benzene | anilines | sibling | 3 |
| ammonia | dioxygen | sibling | 3 |
| ammonia | carbon dioxide | sibling | 3 |
| benzene | cyclopentane | sibling | 2 |
| propene | ammonia | sibling | 2 |
