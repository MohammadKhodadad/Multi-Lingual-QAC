# Wikipedia-name quality check

For a concept mentioned in an English patent, does its Wikipedia title in language L appear in the same patent's L translation?

| lang | name coverage | checks | wiki hit rate | concept present | conditional (wiki\|present) |
|------|---------------|--------|---------------|-----------------|------------------------------|
| zh | 1490/3227 (46%) | 0 | 0/0 (0.0%) | 0.0% | 0.0% |
| de | 1583/3227 (49%) | 2718 | 956/2718 (35.2%) | 42.7% | 82.3% |
| fr | 1520/3227 (47%) | 17310 | 8552/17310 (49.4%) | 54.8% | 90.2% |
| es | 1405/3227 (44%) | 3710 | 2335/3710 (62.9%) | 67.7% | 93.0% |

- **wiki hit rate**: Wikipedia title found in the L translation.
- **concept present**: the concept appears in the L translation by *any* name.
- **conditional**: hit rate among docs where the concept is actually present (isolates name quality from untranslated mentions).

## Most common name mismatches (concept present, Wikipedia title absent)

| lang | concept | Wikipedia title | matched instead | count |
|------|---------|-----------------|-----------------|-------|
| fr | inhibitor | Inhibiteur (chimie) | inhibiteur | 180 |
| fr | chlorpyrifos | Chlorpyriphos-éthyl | Cobalt | 74 |
| fr | silane | Silane (composé) | silane | 61 |
| fr | isocyanic acid | Acide isocyanique | isocyanate | 60 |
| fr | ligand | Ligand (chimie) | ligand | 52 |
| fr | peptide | Peptide | peptides | 48 |
| fr | polypeptide | Polypeptide | polypeptides | 41 |
| fr | agonist | Agoniste (biochimie) | agoniste | 36 |
| fr | hydroxy group | Hydroxyle | hydroxy | 33 |
| fr | cation | Cation | cations | 29 |
| fr | carbon nanotube | Nanotube de carbone | nanotubes de carbone | 27 |
| fr | ligand | Ligand (chimie) | ligands | 24 |
| fr | carboxy group | Carboxyle | carboxy | 23 |
| fr | quartz | Quartz (minéral) | quartz | 22 |
| fr | alpha-particle | Particule α | alpha | 20 |
| fr | fatty acid | Acide gras | acides gras | 19 |
| fr | radical | Radical (chimie) | radical | 18 |
| fr | antagonist | Antagoniste (biochimie) | antagoniste | 18 |
| de | silicon atom | Silicium | Silizium | 17 |
| fr | photon | Photon | gamma | 16 |
| es | hydrocarbon | Hidrocarburo | hidrocarburos | 14 |
| de | polyol | Polyole | polyol | 13 |
| de | titanium atom | Titan (Element) | Titan | 13 |
| fr | glycine | Glycine (acide aminé) | glycine | 13 |
| fr | silicon dioxide | Verre de quartz | SiO2 | 12 |
| fr | biological pigment | Pigment | pigments | 12 |
| fr | tantalum atom | Tantale (chimie) | tantale | 12 |
| fr | carbohydrate | Glucide | saccharide | 12 |
| fr | hydrate | Hydrate | hydrates | 12 |
| fr | pesticide | Pesticide | pesticides | 12 |
| fr | polysaccharide | Polysaccharide | polysaccharides | 11 |
| es | antagonist | Antagonista (bioquímica) | antagonista | 11 |
| fr | polyol | Polyol | polyols | 10 |
| fr | urea | Urée | Carbamide | 9 |
| es | mineral | Mineral | minerales | 9 |
| fr | alkane | Alcane | alcanes | 9 |
| fr | anion | Anion | anions | 9 |
| es | electron | Electrón | beta | 9 |
| fr | Glucagon-like peptide 1 | Glucagon-like peptide-1 | GLP-1 | 9 |
| fr | vitamin (role) | Vitamine | vitamines | 9 |
