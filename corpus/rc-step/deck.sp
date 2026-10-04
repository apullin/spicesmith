Portable RC step response
.param vsupply=1.2
V1 input 0 PULSE(0 1.2 1n 10p 10p 10n 20n)
R1 input output 1k
C1 output 0 1p
.tran 10p 5n 0 10p
* spicesmith-output waveform.txt v(input) v(output)
.save v(input) v(output)
.end
