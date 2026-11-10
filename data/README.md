# Data

`jura.csv` contains the 359 Jura samples (259 prediction-set and 100
validation-set locations combined) used in Sect. 6.2.

| Column | Meaning |
| --- | --- |
| `X`, `Y` | Easting and northing (km, local grid) |
| `Rock` | Rock type, coded 1-5 |
| `Land` | Land use, coded 1-4 (not used) |
| `Cd` ... `Zn` | Topsoil heavy-metal concentrations (mg/kg; not used) |

Rock codes follow Goovaerts (1997); the names used in the paper are
1 Argovian, 2 Kimmeridgian, 3 Sequanian, 4 Portlandian, 5 Quaternary.

**Provenance.** The data were collected by the Swiss Federal Institute of
Technology at Lausanne and first analysed by Atteia et al. (1994). They are
distributed with Goovaerts (1997) and with the `gstat` R package (data sets
`jura.pred` and `jura.val`). They are redistributed here unchanged for
reproducibility; please cite the original sources:

- Atteia O, Dubois JP, Webster R (1994) Geostatistical analysis of soil
  contamination in the Swiss Jura. Environ Pollut 86(3):315-327.
- Goovaerts P (1997) Geostatistics for natural resources evaluation.
  Oxford University Press, New York.
