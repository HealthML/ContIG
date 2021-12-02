# TODO: fill in path to imputed data (should contain .bgen files)
PTH_TO_IMP=PATH/TO/imputed/EGA.../
# TODO: fill in path to imputed data (should contain .sample files) (need to update below as well)
PTH_TO_SMP=PATH/TO/imputed/sample/

DST_PTH=imputed_merged/
mkdir $DST_PTH

MLOCAL=.
PTH_TO_SUBSET=$MLOCAL/iids_with_retinal_scan.txt

THREADS=40

# convert bgen to pgen and remove duplicates
for CHROMO in {1..22}; do

    PTH_TO_OUT=$DST_PTH/ukb_imp_chr${CHROMO}
    # TODO: need to update the sample file to include your reference number
    plink2 \
        --bgen ${PTH_TO_IMP}/ukb_imp_chr${CHROMO}_v3.bgen ref-first \
        --sample ${PTH_TO_SMP}/YOURSAMPLEFILE${CHROMO}.sample \
        --keep $PTH_TO_SUBSET \
        --rm-dup force-first \
        --out $PTH_TO_OUT \
        --threads $THREADS \
done

# merge all chromo pgens into one big pgen
PMERGE_LIST=${MLOCAL}/pmerge_list.txt
PGEN_MERGED=$DST_PTH/ukb_imp_allchr
plink2 \
    --pmerge-list $PMERGE_LIST \
    --make-pgen \
    --out $PGEN_MERGED \
    --threads $THREADS

# clean up individual pgens
ls ${DST_PTH}/ukb_imp_chr{1..22}.(log|pvar|psam|pgen)
rm -I ${DST_PTH}/ukb_imp_chr{1..22}.(log|pvar|psam|pgen)
