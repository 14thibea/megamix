if [ `id -u` -ne 0 ]; then 
  export MODULEPATH=/cm/local/modulefiles:/cm/shared/modulefiles
  export PATH=${PATH}:/sbin:/usr/sbin
else 
  export MODULEPATH=/cm/local/modulefiles
fi
#----------------------------------------------------------------------#
# system-wide profile.modules                                          #
# Initialize modules for all sh-derivative shells                      #
#----------------------------------------------------------------------#
trap "" 1 2 3

case "$0" in
    -bash|bash|*/bash) . /cm/local/apps/environment-modules/3.2.6//Modules/default/init/bash; export -f module ;; 
       -ksh|ksh|*/ksh) . /cm/local/apps/environment-modules/3.2.6//Modules/default/init/ksh ;; 
          -sh|sh|*/sh) . /cm/local/apps/environment-modules/3.2.6//Modules/default/init/sh; export -f module ;; 
                    *) . /cm/local/apps/environment-modules/3.2.6//Modules/default/init/sh ;; 	# default for scripts
esac

trap 1 2 3
