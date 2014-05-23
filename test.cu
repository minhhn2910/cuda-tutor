#include <stdio.h>
__global__ void kernel( void ) {}


int main( void )
{
	kernel<<< 1, 1 >>>(); //kernel call 

	printf( "Hello, World!\n" );

	return 0;
}

