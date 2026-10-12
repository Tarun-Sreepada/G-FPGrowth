#!/usr/bin/env bash
# Download the SPMF frequent-itemset-mining benchmark datasets into datasets/.
set -euo pipefail
cd "$(dirname "$0")/../datasets"
base=https://www.philippe-fournier-viger.com/spmf/publicdatasets
for f in retail chess mushrooms T10I4D100K foodmartFIM BMS1_itemset_mining BMS2_itemset_mining connect pumsb accidents; do
  [ -s "$f.txt" ] || curl -fsSL -O "$base/$f.txt"
done
