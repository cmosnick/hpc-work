/* Shared Memory example

*/

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <unistd.h>
#include <cstdlib>
 
#include <iostream>


int main(int argc, char * argv[])
{

	pid_t pid;			// process identifier, pid_t is a process id type defined in sys/types

	// =============================
	// BEGIN: Do the Semaphore Setup
	// The semaphore will be a mutex
	// =============================
	int semId; 			// ID of semaphore set
	key_t semKey = 123459; 		// key to pass to semget(), key_t is an IPC key type defined in sys/types
	int semFlag = IPC_CREAT | 0666; // Flag to create with rw permissions

	int semCount = 1; 		// number of semaphores to pass to semget()
	int numOps = 1; 		// number of operations to do
	
	// Create the semaphore
	// The return value is the semaphore set identifier
	// The flag is a or'd values of CREATE and ugo permission of RW, 
	//			just like used for creating a file
   	if ((semId = semget(semKey, semCount, semFlag)) == -1)
   	{
   		std::cerr << "Failed to semget(" << semKey << "," << semCount << "," << semFlag << ")" << std::endl;
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

	argument.val = 1; // NOTE: We are setting this to one to make it a MUTEX
	if( semctl(semId, 0, SETVAL, argument) < 0)
	{
		std::cerr << "Init: Failed to initialize (" << semId << ")" << std::endl; 
		exit(1);
	}
	else
	{
		std::cout << "Init: Initialized (" << semId << ")" << std::endl; 
	}

	// =============================
	// END: Do the Semaphore Setup
	// =============================

	// =========================================
	// BEGIN: Do the Shared Memory Segment Setup
	// 
	// =========================================

	int shmId; 			// ID of shared memory segment
	key_t shmKey = 123460; 		// key to pass to shmget(), key_t is an IPC key type defined in sys/types
	int shmFlag = IPC_CREAT | 0666; // Flag to create with rw permissions
	
	// This will be shared:
	const unsigned int targetFibNum = 25;
	unsigned long * shm;
	unsigned long * sharedIndexPtr = NULL;
	
	// (from shm.h)
	// int shmget(key_t key, size_t size, int shmflg);
	//
	// shmget() returns the identifier of the shared memory segment associated with the value of the argument key.
	// A new shared memory segment, with size equal to the value of size rounded up to a multiple of PAGE_SIZE, is
	// created  if  key has the value IPC_PRIVATE or key isn't IPC_PRIVATE, no shared memory segment corresponding
	// to key exists, and IPC_CREAT is specified in shmflg.
	// 
	// If shmflg specifies both IPC_CREAT and IPC_EXCL and a shared memory segment already exists  for  key,  then
	// shmget()  fails  with  errno  set to EEXIST.  (This is analogous to the effect of the combination O_CREAT |
	// O_EXCL for open(2).)

	// targetFibNum + 2, the targetFibNum+1 element will be a counter
	if ((shmId = shmget(shmKey, (targetFibNum+2) * sizeof(unsigned long), shmFlag)) < 0)
	{
		std::cerr << "Init: Failed to initialize shared memory (" << shmId << ")" << std::endl; 
		exit(1);
	}
		
	// (from shm.h)
	// void *shmat(int shmid, const void *shmaddr, int shmflg);
	//
	// shmat() attaches the shared memory segment identified by shmid to the address space of the calling process.
	// The attaching address is specified by shmaddr with one of the following criteria:
	// 
 	// If shmaddr is NULL, the system chooses a suitable (unused) address at which to attach the segment.
	// 
	// If shmaddr isn't NULL and SHM_RND is specified in shmflg, the attach occurs at the address equal to shmaddr
	// rounded  down to the nearest multiple of SHMLBA.  Otherwise shmaddr must be a page-aligned address at which
	// the attach occurs.
	// 
	// If SHM_RDONLY is specified in shmflg, the segment is attached for reading and the process  must  have  read
	// permission for the segment.  Otherwise the segment is attached for read and write and the process must have
	// read and write permission for the segment.  There is no notion of a write-only shared memory segment.

	if ((shm = (unsigned long *)shmat(shmId, NULL, 0)) == (unsigned long *) -1)
	{
		std::cerr << "Init: Failed to attach shared memory (" << shmId << ")" << std::endl; 
		exit(1);
	}
	
	// =========================================
	// END: Do the Shared Memory Segment Setup
	// =========================================

	// Initialize our fibonacci sequence generator
	// i[0 1 2 3 4 5 6  7  8 ...   N ]
	// f[0 1 1 2 3 5 8 13 21 ... f(N)]
	shm[0] = 0;
	shm[1] = 1;
	sharedIndexPtr = &shm[targetFibNum+1];	// our pointer to hold the array shared index variable
	*sharedIndexPtr = 2;			// init to two because of the [0] and [1] are known
	unsigned long lastIndexProcessed = 1;	// use this variable to track what a process just did
		
	// ===================================
	// This should now look familiar
	// 	we will just play a different 
	//	game in the processes
	// ===================================


	pid = fork();	// fork, which replicates the process 
	
	// +++++++++++++++++++++++++++++++++++++++
	//NOTE: for this shared memory segment... 
	// 	the value of the shm variable is 
	//	an address of shared memory that
	// +++++++++++++++++++++++++++++++++++++++

	if ( pid < 0 )
	{ 
		std::cerr << "Could not fork!!! ("<< pid <<")" << std::endl;
		exit(1);
	}
	
	std::cout << "I just forked without error, I see ("<< pid <<")" << std::endl;
	
	if ( pid == 0 ) // Child process 
	{
		std::cout << "In the child (if): " << std::endl; 
		
		while (lastIndexProcessed < targetFibNum)
		{
			// Get the semaphore (P-op)
			struct sembuf operations[1];
			operations[0].sem_num = 0; 	// use the first(only, because of semCount above) semaphore
			operations[0].sem_flg = 0;	// allow the calling process to block and wait

			// Set up the sembuf structure.
			operations[0].sem_op = -1; 	// this is the operation... the value is added to semaphore (a P-Op = -1)

			std::cout << "In the child (if): about to blocking wait on semaphore" << std::endl; 
			int retval = semop(semId, operations, numOps);

			if(0 == retval)
			{
				// Compute the next number
				const unsigned long index = *sharedIndexPtr;

				std::cout << "In the child (if): Successful P-operation on (" << semId << "), index=" << index << std::endl; 

				if (index <= targetFibNum)
				{
					std::cout << "In the child (if): computing index (" << index << ")" ; 
					shm[index] = shm[index - 1] + shm[index-2];
					std::cout << "as (" << shm[index] << ")" << std::endl; 
				}
				*sharedIndexPtr += 1;
				// mark the last index processed
				lastIndexProcessed = index;
				sleep(1);
			}
			else
			{
				std::cerr << "In the child (if): Failed P-operation on (" << semId << ")" << std::endl; 
				_exit(1);
			}

		
			// Release the semaphore (V-op)
			operations[0].sem_op = 1; 	// this the operation... the value is added to semaphore (a V-Op = 1)
		
			std::cout << "In the child (if): about to release semaphore" << std::endl; 
			retval = semop(semId, operations, numOps);
			if(0 == retval)
			{
				std::cout << "In the child (if): Successful V-operation on (" << semId << ")" << std::endl; 
			}
			else
			{
				std::cerr << "In the child (if): Failed V-operation on (" << semId << ")" << std::endl; 
			}
		
		} // END of while we have not computed the full sequence
		
		_exit(0);
	} 
	else		// Parent Process
	{
		std::cout << "In the parent (if-else): " << std::endl; 
		
		while (lastIndexProcessed <= targetFibNum)
		{
			// Get the semaphore (P-op)
			struct sembuf operations[1];
			operations[0].sem_num = 0; 	// use the first(only, because of semCount above) semaphore
			operations[0].sem_flg = 0;	// allow the calling process to block and wait

			// Set up the sembuf structure.
			operations[0].sem_op = -1; 	// this is the operation... the value is added to semaphore (a P-Op = -1)

			std::cout << "In the parent (if-else): about to blocking wait on semaphore" << std::endl; 
			int retval = semop(semId, operations, numOps);

			if(0 == retval)
			{
				// Compute the next number
				const unsigned long index = *sharedIndexPtr;
				
				std::cout << "In the parent (if-else): Successful P-operation on (" << semId << "), index=" << index << std::endl; 

				if (index <= targetFibNum)
				{
					std::cout << "In the parent (if-else): computing index (" << index << ")" ; 
					shm[index] = shm[index - 1] + shm[index-2];
					std::cout << "as (" << shm[index] << ")" << std::endl; 
				}
				*sharedIndexPtr += 1;
				// mark the last index processed
				lastIndexProcessed = index;
				sleep(1);

			}
			else
			{
				std::cerr << "In the parent (if-else): Failed P-operation on (" << semId << ")" << std::endl; 
				_exit(1);
			}

		
			// Release the semaphore (V-op)
			operations[0].sem_op = 1; 	// this the operation... the value is added to semaphore (a V-Op = 1)
		
			std::cout << "In the parent (if-else): about to release semaphore" << std::endl; 
			retval = semop(semId, operations, numOps);
			if(0 == retval)
			{
				std::cout << "In the parent (if-else): Successful V-operation on (" << semId << ")" << std::endl; 
			}
			else
			{
				std::cerr << "In the parent (if-else): Failed V-operation on (" << semId << ")" << std::endl; 
			}
			
		} // END of while we have not computed the full sequence
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
	
	const int locationIndex = targetFibNum;
	std::cout << "Computed Fib(" << targetFibNum << ") = " << shm[locationIndex] << std::endl;
	std::cout << "CHECK : Fib(10)  = 55" << std::endl;
	std::cout << "CHECK : Fib(25)  = 75025" << std::endl;
	std::cout << "CHECK : Fib(50)  = 12586269025" << std::endl;
	std::cout << "CHECK : Fib(100) = 354224848179261915075" << std::endl;
	
	return 0;
}

