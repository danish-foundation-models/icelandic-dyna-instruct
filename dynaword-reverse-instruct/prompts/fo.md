# Uppgáva

Tú gert føroyskt venjingartilfar til eitt málmodell.

Les tekstin og skriva ein realistiskan og sjálvstøðugan brúkarafyrispurning, sum teksturin er eitt gott svar til. Met eisini, um teksturin er nóg góður til venjingartilfar.

Svar bert við JSON, sum fylgir kravda skemanum.

# Møgulig øki

- `encyclopedic`: ein neutral frágreiðing, eitt yvirlit ella ein uppslagsgrein
- `news_report`: tíðindi, hagtøl ella ein saklig frágreiðing um eina gongd
- `public_guidance`: almenn kunning ella praktisk vegleiðing
- `opinion`: ein viðmerking, kjakgrein ella grundgivin áskoðan
- `instructional`: undirvísing, vegleiðing ella ein stigvís frágreiðing

# Krøv

- Fyrispurningurin skal vera á natúrligum føroyskum.
- Hann skal kunna standa einsamallur.
- Fyrispurningurin skal vera í mesta lagi 300 tekn.
- Hann má ikki nevna tekstin, brotið omanfyri, kelduna ella dátusavnið.
- Hann skal samsvara við innihald, stíl og endamál í tekstinum.
- Hann má ikki biðja um upplýsingar ella eitt format, sum teksturin ikki gevur.
- Tak neyðugar detaljur við, men endurnýt ikki langar orðingar úr tekstinum.
- Vraka ósamanhangandi tekstbrot, yvirskriftir uttan innihald og málsliga vánaligar tekstir.
- Vraka tekstir við viðkvæmum persónsupplýsingum ella óskiljandi OCR- og ASR-villum.
- Orsøkin skal vera stutt og í mesta lagi 300 tekn.

# Keldusamanhangur

Kelda: $source

Møguligt øki: $domain_hint

Frágreiðing um kelduna: $source_context

# Tekstur

$passage

# Úttak

Set `accept` til `false`, `domain` og `instruction` til `null`, og gev eina stutta orsøk, um teksturin ikki er egnaður.

Set annars `accept` til `true`, vel eitt av loyvdu økjunum og skriva fyrispurningin.
