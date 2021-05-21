set term gif size 1600,1600 animate delay 5 loop 0 optimize
set output "anim4d_vector_field.gif"


set xrange [0:25]
set yrange [0:25]
set zrange [0:25]

set view equal xyz
set ticslevel 0

set border 4095

# set view 90,0,1,1

do for [i=1:400]{
    splot 'data4d.txt' index i u 1:2:3:($4*100):($5*100):($6*100) w vectors lw 1
}
