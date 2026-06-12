# Confusion analysis — does a wrong (look-alike) compound beat the right one?

Confusion rate = fraction of queries where a hard-negative (chemically-similar wrong compound) ranks above every gold document.

| model | de | en | es | fr | zh | ALL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | 11.1% (n=27) | 7.4% (n=27) | 8.0% (n=25) | 11.1% (n=27) | 26.9% (n=26) | 12.9% (n=132) |
| `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | 37.0% (n=27) | 25.9% (n=27) | 28.0% (n=25) | 22.2% (n=27) | 34.6% (n=26) | 29.5% (n=132) |
| `google/embeddinggemma-300m` | 11.1% (n=27) | 7.4% (n=27) | 0.0% (n=25) | 11.1% (n=27) | 3.9% (n=26) | 6.8% (n=132) |
| `ibm-granite/granite-embedding-278m-multilingual` | 11.1% (n=27) | 18.5% (n=27) | 8.0% (n=25) | 22.2% (n=27) | 30.8% (n=26) | 18.2% (n=132) |
| `intfloat/multilingual-e5-large-instruct` | 29.6% (n=27) | 18.5% (n=27) | 40.0% (n=25) | 22.2% (n=27) | 30.8% (n=26) | 28.0% (n=132) |
| `nomic-ai/nomic-embed-text-v2-moe` | 7.4% (n=27) | 14.8% (n=27) | 0.0% (n=25) | 11.1% (n=27) | 19.2% (n=26) | 10.6% (n=132) |
| `Qwen/Qwen3-Embedding-0.6B` | 18.5% (n=27) | 11.1% (n=27) | 4.0% (n=25) | 11.1% (n=27) | 19.2% (n=26) | 12.9% (n=132) |
| `sentence-transformers/LaBSE` | 25.9% (n=27) | 25.9% (n=27) | 20.0% (n=25) | 25.9% (n=27) | 26.9% (n=26) | 25.0% (n=132) |

## Most frequent confusions (winning look-alike, all models)

| right compound | beaten by (look-alike) | relation | count |
| --- | --- | --- | ---: |
| protein polypeptide chain | polypeptide | parent | 21 |
| propene | ethene | sibling | 18 |
| ethyl | methyl | sibling | 17 |
| benzene | biphenyl | sibling | 13 |
| polysulfur | elemental sulfur | parent | 11 |
| sulfide(2-) | sulfate | sibling | 8 |
| methacrylic acid | acrylic acid | sibling | 8 |
| poly(vinyl alcohol) macromolecule | poly(alkylene) macromolecule | sibling | 8 |
| adipic acid | succinic acid | sibling | 6 |
| poly(propylene glycol) macromolecule | poly(ethylene glycol) | sibling | 6 |
| sulfide(2-) | sulfite | sibling | 5 |
| propene | dioxygen | sibling | 3 |
| carbon dioxide | dioxygen | sibling | 3 |
| benzene | anilines | sibling | 3 |
| ammonia | dioxygen | sibling | 3 |
| poly(propylene glycol) macromolecule | poly(ether) macromolecule | parent | 3 |
| poly(vinyl alcohol) macromolecule | polyurethane macromolecule | sibling | 3 |
| adipic acid | sebacic acid | sibling | 3 |
| ammonia | carbon dioxide | sibling | 3 |
| sulfite | sulfonate | sibling | 2 |
| propene | carbon dioxide | sibling | 2 |
| phenol | Trolox | sibling | 2 |
| sulfide(2-) | thiosulfate(2-) | sibling | 2 |
| sulfite | sulfate | sibling | 2 |
| polyurethane macromolecule | poly(vinyl alcohol) macromolecule | sibling | 2 |
| phenol | bisphenol | sibling | 2 |
| propene | methane | sibling | 2 |
| phenol | polyphenol | sibling | 2 |
| sulfide(2-) | hydrogenphosphate | sibling | 1 |
| benzene | acetonitrile | sibling | 1 |
