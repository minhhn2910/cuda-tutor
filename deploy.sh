#!/bin/bash
echo "How many students?"
read num
	for((k=0;k<num;k++))
	do
		usrnum=""
		if [ $k -lt 10 ]
		then
			usrnum="0"$k
		else
			usrnum=$k
		fi			
		username="std$usrnum"
		echo "deploying for $username"
		result=(scp * $username:~/cuda)
	done
