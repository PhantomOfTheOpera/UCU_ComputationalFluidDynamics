set term gif size 1600,1600 animate delay 5 loop 0 optimize
set output "anim4d_vector_field.gif"


set xrange [0:27]
set yrange [0:27]
set zrange [0:27]

set view equal xyz
set ticslevel 0

set border 4095

# set view 90,0,1,1

do for [i=1:30]{
    splot 'data4d.txt' index i u 1:2:3:($4*20):($5*20):($6*20) w vectors lw 1
}
