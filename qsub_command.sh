echo
". /home/ethibeau-sutre/commands/modules.sh;
export PATH='/cm/shared/apps/python3-anaconda-updated/bin:/cm/shared/apps/python3-anaconda-updated/lib/python3.5/site-packages:$PATH';
export PYTHONPATH='/home/ethibeau-sutre/Scripts:$PYTHONPATH';
export OMP_NUM_THREADS=1;
export MKL_NUM_THREADS=1;
python /home/ethibeau-sutre/Scripts/VBGMM2.py" | qsub -e /home/ethibeau-sutre -o /home/ethibeau-sutre
