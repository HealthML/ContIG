# utility for downloading and preprocessing Polygenic risk scores

Utilities to download PGS catalog data, and then apply those weights to compute polygenic risk scores for individuals in the UK Biobank. Only works for PGS files that provide rsids (instead of only chromo/bp), but this is most of the PGS catalog data.

## Download PGS Catalog scores

use the functionality in `download_pgs.py`: 
```python
download_multiple_pgs('all')
# or e.g.
download_multiple_pgs(['PGS000001', 'PGS000002'])
```

This will download all scoring files and metadata to `DST` (defaults to `pgs_data` in the working directory).


## Apply risk scores on genetic data

### Using imputed data

This will first convert the UKB `bgen` files into plink `pgen` files, then join all chromosomes into one big file, and then run `plink2 --score` on this. There might be a slightly better way using e.g. `PRSIce2` or `LDpred2` instead of `plink2`.

To merge and convert the `bgen`s use `merge_bgen_to_pgen.sh`. First, replace `iids_with_retinal_scan.txt` with a list of your iids with retinal fundus images. If you change the `MLOCAL` variable, you also need to change the paths in the `pmerge_list.txt` files. You will also need to update the paths to your `.bgen` and `.sample` files. Finally, run the script.

Next, run the `create_pgs.sh` script. You need to be in an environment with python and pandas installed. If you're using a newer version of the PGS catalog, you might need to update the number of PGS scores to include the most current number.

Note that some of the PGS Catalog files will be skipped because they only provide weights based on chromosome-positions, instead of rsids; plink2 only works with rsids so far.


### Using only microarray data
You can run the analysis on unimputed microarray data (`bed`-files) only: you only need to modify a couple of commands in the two `.sh` files and won't need to convert away from `bed` files (but still need to merge them, afaik).

Note: this should give reasonable results but will only use a small fraction of the actual SNPs provided by PGS catalog. It's going to be a lot faster and more disk-space efficient than with imputed data, though.


