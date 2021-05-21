set term gif size 1600,1600 animate delay 5 loop 0 optimize
set output "anim.gif"


set xrange [0:120]
set yrange [0:120]
set zrange [-1000:4500]

set style fill  transparent solid 0.70 border
set pm3d depthorder border linecolor rgb "#a0a0f0"  linewidth 0.5

set border 4095

do for [i=1:40]{
    splot 'data4d.txt' index i u 1:2:3:4 w pm3d
}
