# Datasets

Transaction databases for frequent itemset mining from the
[SPMF open-source data mining library](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php#d2)
(Philippe Fournier-Viger et al.), section *frequent itemset mining*. One transaction per line,
items are space separated integers. The data files are not committed; fetch them with
`scripts/get_datasets.sh`.

| file | source |
|---|---|
| `retail.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/retail.txt |
| `chess.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/chess.txt |
| `mushrooms.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/mushrooms.txt |
| `T10I4D100K.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/T10I4D100K.txt |
| `foodmartFIM.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/foodmartFIM.txt |
| `BMS1_itemset_mining.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/BMS1_itemset_mining.txt |
| `BMS2_itemset_mining.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/BMS2_itemset_mining.txt |
| `connect.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/connect.txt |
| `pumsb.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/pumsb.txt |
| `accidents.txt` | https://www.philippe-fournier-viger.com/spmf/publicdatasets/accidents.txt |

Original provenance (see the SPMF page): chess/connect/mushroom/pumsb/accidents come from the
UCI repository and the FIMI repository (http://fimi.uantwerpen.be/data/), T10I4D100K is
generated with the IBM Quest generator, retail is from
Brijs et al. (KDD 1999), BMS1/BMS2 are from KDD-Cup 2000 (Blue Martini Software).
