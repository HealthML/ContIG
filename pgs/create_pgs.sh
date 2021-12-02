
MLOCAL=.

THREADS=40
DST_PTH=imputed_merged/

OUT_DIR=precomputed_pgs
mkdir $OUT_DIR

PGEN_MERGED=$DST_PTH/ukb_imp_allchr
# TODO: update PGS ID numbers if you have more
for i in {000001..000920}; do
    PGS_FN=pgs_data/PGS${i}/ScoringFiles/PGS${i}.txt.gz
    echo "starting PGS${i}"
    if test -f "$PGS_FN"; then
        # don't look here! too lazy to find a way to do this in bash...
        PYTHON_LOOKUP="import pandas as pd; cols = [c.lower() for c in pd.read_csv('$PGS_FN', comment='#', sep='\t', nrows=0).columns]; print('rsid') if 'rsid' in cols else print('pos'); print(cols.index('rsid')+1 if 'rsid' in cols else 0); print(cols.index('effect_allele')+1 if 'effect_allele' in cols else 0); print(cols.index('effect_weight')+1 if 'effect_weight' in cols else 0);"

        OUTPUT=()
        while read line; do
            OUTPUT+=($line)
        done < <(python -c $PYTHON_LOOKUP)
        MODE=${OUTPUT[1]}

        OUT_FN=$OUT_DIR/PGS${i}
        if [ "$MODE" = "rsid" ]; then
            RSID_COL=${OUTPUT[2]}
            ALLELE_COL=${OUTPUT[3]}
            WEIGHT_COL=${OUTPUT[4]}
            echo "using rsid mode, columns $RSID_COL $ALLELE_COL $WEIGHT_COL"
            plink2 \
                --pfile $PGEN_MERGED \
                --score $PGS_FN $RSID_COL $ALLELE_COL $WEIGHT_COL \
                --threads $THREADS \
                --out $OUT_FN
        else
            # some of the PGS files only include chromosome positions; plink2 doesn't seem to support this at the moment
            echo "no rsids, only positions - skipping for now"
        fi
    else
        echo "$PGS_FN doesn't exist"
    fi
done
