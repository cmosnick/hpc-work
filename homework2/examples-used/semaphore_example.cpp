/* Semaphore example

(from sys/sem.h)

A semaphore shall be represented by an anonymous structure containing the following members:
      unsigned short  semval   Semaphore value.
      pid_t           sempid   Process ID of last operation.
      unsigned short  semncnt  Number of processes waiting for semval
                               to become greater than current value.
      unsigned short  semzcnt  Number of processes waiting for semval
                               to become 0.


The semid_ds structure shall contain the following members:
      struct ipc_perm  sem_perm  Operation permission structure.
      unsigned short   sem_nsems Number of semaphores in set.
      time_t           sem_otime Last semop
       () time.
      time_t           sem_ctime Last time changed by semctl
       ().

*/

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <unistd.h>
#include <cstdlib>
 
#include <iostream>


int main(int argc, char * argv[])
{
	pid_t pid;			// process identifier, pid_t is a process id type defined in sys/types
	int semId; 			// ID of semaphore set
	key_t key = 123456; 		// key to pass to semget(), key_t is an IPC key type defined in sys/types
	int semFlag = IPC_CREAT | 0666; // Flag to create with rw permissions


	int semCount = 1; 		// number of semaphores to pass to semget()
	int numOps = 1; 		// number of operations to do
	
	// Create the semaphore
	// The return value is the semaphore set identifier
	// The flag is a or'd values of CREATE and ugo permission of RW, 
	//			just like used for creating a file
   	if ((semId = semget(key, semCount, semFlag)) == -1)
   	{
   		std::cerr << "Failed to semget(" << key << "," << semCount << "," << semFlag << ")" << std::endl;
		exit(1);
	}
	else
	{
   		std::cout << "Successful semget resulted in (" << semId << std::endl;
	}

	// Initialize the semaphore
	union semun {
		int val;
		struct semid_ds *buf;
		ushort * array;
	} argument;

	argument.val = 0;
	if( semctl(semId, 0, SETVAL, argument) < 0)
	{
		std::cerr << "Init: Failed to initialize (" << semId << ")" << std::endl; 
		exit(1);
	}
	else
	{
		std::cout << "Init: Initialized (" << semId << ")" << std::endl; 
	}

	// ============================== 
	// This should now look familiar
	// ============================== 

	pid = fork();	// fork, which replicates the process 

	if ( pid < 0 )
	{ 
		std::cerr << "Could not fork!!! ("<< pid <<")" << std::endl;
		exit(1);
	}
	
	std::cout << "I just forked without error, I see ("<< pid <<")" << std::endl;
	
	if ( pid == 0 ) // Child process 
	{
		std::cout << "In the child (if): " << std::endl; 
		
		// sembuf defined in sem.h
		//  The sembuf structure shall contain the following members:
		//    unsigned short  sem_num   Semaphore number.
		//    short           sem_op    Semaphore operation.
		//    short           sem_flg   Operation flags.
		//
		// This becomes a pointer to sem-operations that are performed
		// Recall from your reading, googling, or an OS class the two semaphore
		//	operations?
		//	V = signal / increment
		//	P = wait / decrement
		struct sembuf operations[1];

		// Set up the sembuf structure.
		operations[0].sem_num = 0; 	// use the first(only, because of semCount above) semaphore
		operations[0].sem_op = -1; 	// this the operation... the value is added to semaphore (a P-Op = -1)
		operations[0].sem_flg = 0;	// set to 0 to allow the calling process to block and wait

		// semop performs the operations on the semaphore:
		// (from sem.h)
		// int semop(int semid, struct sembuf *sops, unsigned nsops);
		//
		// Each semaphore in a semaphore set has the following associated values:
		//
		//	unsigned short  semval;   /  semaphore value /
		//	unsigned short  semzcnt;  / # waiting for zero /
		//	unsigned short  semncnt;  / # waiting for increase /
		//	pid_t           sempid;   / process that did last op /
		//
		// semop()  performs operations on selected semaphores in the set indicated by semid.  Each of the nsops ele‐
		// ments in the array pointed to by sops specifies an operation to be performed on a single  semaphore.

		std::cout << "In the child (if): about to blocking wait on semaphore" << std::endl; 
		int retval = semop(semId, operations, numOps);

		if(0 == retval)
		{
			std::cout << "In the child (if): Successful P-operation on (" << semId << ")" << std::endl; 
			_exit(0);
		}
		else
		{
			std::cerr << "In the child (if): Failed P-operation on (" << semId << ")" << std::endl; 
			_exit(1);
		}

	} 
	else		// Parent Process
	{
		std::cout << "In the parent (if-else): " << std::endl; 

		// pause for a little bit?
		const unsigned int sleepNumber = 5;
		std::cout << "In the parent (if-else): my sleep number is " << sleepNumber << std::endl; 		
		sleep(sleepNumber);
		std::cout << "In the parent (if-else): the dragon has awoken" << std::endl; 		
		

		// See comments above
		struct sembuf operations[1];

		// see comments above
		operations[0].sem_num = 0;
		operations[0].sem_op = 1;	// this the operation... the value is added to semaphore (a V-Op = +1)
		operations[0].sem_flg = 0;

		// see comments above
		std::cout << "In the parent (if-else): about to V-op on semaphore" << std::endl; 
		int retval = semop(semId, operations, numOps);

		if(0 == retval)
		{
			std::cout << "In the parent (if-else): Successful V-operation on (" << semId << ")" << std::endl; 
		}
		else
		{
			std::cerr << "In the parent (if-else): Failed V-operation on (" << semId << ")" << std::endl; 
		}


	}
	

	// ============================== 
	// All this code is boiler-plate	
	// ============================== 

	std::cout << "In the parent: " << std::endl; 

	int status;	// catch the status of the child

	do  // in reality, mulptiple signals or exit status could come from the child
	{

		pid_t w = waitpid(pid, &status, WUNTRACED | WCONTINUED);
		if (w == -1)
		{
			std::cerr << "Error waiting for child process ("<< pid <<")" << std::endl;
			break;
		}
		
		if (WIFEXITED(status))
		{
			if (status > 0)
			{
				std::cerr << "Child process ("<< pid <<") exited with non-zero status of " << WEXITSTATUS(status) << std::endl;
				continue;
			}
			else
			{
				std::cout << "Child process ("<< pid <<") exited with status of " << WEXITSTATUS(status) << std::endl;
				continue;
			}
		}
		else if (WIFSIGNALED(status))
		{
			std::cout << "Child process ("<< pid <<") killed by signal (" << WTERMSIG(status) << ")" << std::endl;
			continue;			
		}
		else if (WIFSTOPPED(status))
		{
			std::cout << "Child process ("<< pid <<") stopped by signal (" << WSTOPSIG(status) << ")" << std::endl;
			continue;			
		}
		else if (WIFCONTINUED(status))
		{
			std::cout << "Child process ("<< pid <<") continued" << std::endl;
			continue;
		}
	}
	while (!WIFEXITED(status) && !WIFSIGNALED(status));
	
	
	return 0;
}

