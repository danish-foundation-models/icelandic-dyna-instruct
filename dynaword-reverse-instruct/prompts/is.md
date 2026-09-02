# Verkefni

Þú býrð til íslensk þjálfunargögn fyrir mállíkan.

Lestu textann og skrifaðu raunhæf, sjálfstæð fyrirmæli frá notanda sem textinn er gott svar við. Met einnig hvort textinn sé nógu góður sem þjálfunargögn.

Skilaðu aðeins JSON sem fylgir uppgefna skemanu.

# Mögulegir flokkar

- `encyclopedic`: hlutlaus útskýring, yfirlit eða uppflettigrein
- `judicial`: dómsmál, lagaleg röksemdafærsla eða hlutlaus lýsing á réttarmáli
- `legislation`: lög, reglugerð, frumvarp eða opinber regla
- `parliamentary`: þingræða, opinber ræða eða röksemdafærsla um stefnumál
- `opinion`: skoðun, pistill eða rökstudd gagnrýni
- `instructional`: kennsla, leiðbeiningar eða útskýring í skrefum
- `literary`: bókmenntatexti, frásögn eða sögulegur stíll

# Kröfur

- Fyrirmælin skulu vera á eðlilegri íslensku.
- Þau skulu geta staðið ein og sér.
- Fyrirmælin skulu vera að hámarki 300 stafir.
- Þau mega ekki minnast á textann, kaflann hér að ofan, heimildina eða gagnasafnið.
- Þau skulu passa við efni, stíl og tilgang textans.
- Þau mega ekki biðja um upplýsingar eða framsetningu sem textinn veitir ekki.
- Hafðu nægar efnisupplýsingar með en ekki afrita langa orðalagskafla úr textanum.
- Hafnaðu ósamfelldum brotum, fyrirsögnum án efnis og illa unnum texta.
- Hafnaðu texta með viðkvæmum persónuupplýsingum eða ónothæfum OCR- og ASR-villum.
- Lagatexti má ekki vera settur fram sem persónusniðin lögfræðiráðgjöf.
- Ástæðan skal vera stutt og að hámarki 300 stafir.

# Heimildarsamhengi

Heimild: $source

Líklegur flokkur: $domain_hint

Lýsing á heimild: $source_context

# Texti

$passage

# Úttak

Settu `accept` á `false`, `domain` og `instruction` á `null`, og gefðu stutta ástæðu ef textinn er ekki nothæfur.

Annars skaltu setja `accept` á `true`, velja einn leyfilegan flokk og skrifa fyrirmælin.
