set term gif size 1600,1600 animate delay 25 loop 0 optimize
set output "anim5d.gif"


set xrange [0:27]
set yrange [0:27]
set zrange [0:27]

set view equal xyz
set ticslevel 0

set style fill  transparent solid 0.70 border
set border 4095
# set palette defined (-1 "red", 0 "green", 1 "blue")

do for [i=1:20]{
    splot 'data5d.txt' index i u 1:2:3:4 w pm3d
}
