#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <iterator>
#include <stdio.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

using namespace std;
using namespace cv;

#define	TIME_MEM_KEY	99		/* kind of like a port number */
#define	TIME_SEM_KEY	9901		/* like a filename	      */
#define oops(m,x)  { perror(m); exit(x); }
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define	SEG_SIZE	((size_t)FRAME_WIDTH*FRAME_HEIGHT*3*sizeof(int)*2)		/* size of segment	*/
union semun { int val ; struct semid_ds *buf ; ushort *array; };

// yuv
struct Picture
{
	unsigned char *data[4];
	int stride[4];
};
typedef struct Picture Picture;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );

string cascadeName = "./haarcascade_frontalface_alt.xml";
string nestedCascadeName = "./haarcascade_eye.xml";

/*
 * build and execute a 2-element action set:
 *    wait for 0 on n_writers AND increment n_readers
 */
void wait_and_lock( int semset_id )
{
	union semun   sem_info;         /* some properties      */
	struct sembuf actions[2];	/* action set		*/

	actions[0].sem_num = 1;		/* sem[1] is n_writers	*/
	actions[0].sem_flg = SEM_UNDO;	/* auto cleanup		*/
	actions[0].sem_op  = 0 ;	/* wait for 0		*/

	actions[1].sem_num = 0;		/* sem[0] is n_readers	*/
	actions[1].sem_flg = SEM_UNDO;	/* auto cleanup		*/
	actions[1].sem_op  = +1 ;	/* incr n_readers	*/

	if ( semop( semset_id, actions, 2) == -1 )
		oops("semop: locking", 10);
}

/*
 * build and execute a 1-element action set:
 *    decrement num_readers
 */
void release_lock( int semset_id )
{
	union semun   sem_info;         /* some properties      */
	struct sembuf actions[1];	/* action set		*/

	actions[0].sem_num = 0;		/* sem[0] is n_readers	*/
	actions[0].sem_flg = SEM_UNDO;	/* auto cleanup		*/
	actions[0].sem_op  = -1 ;	/* decr reader count	*/

	if ( semop( semset_id, actions, 1) == -1 )
		oops("semop: unlocking", 10);
}

int main( int argc, const char** argv )
{
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    const string scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    const string nestedCascadeOpt = "--nested-cascade";
    size_t nestedCascadeOptLen = nestedCascadeOpt.length();
    const string tryFlipOpt = "--try-flip";
    size_t tryFlipOptLen = tryFlipOpt.length();
    string inputName;
    bool tryflip = false;

    help();

    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( nestedCascadeOpt.compare( 0, nestedCascadeOptLen, argv[i], nestedCascadeOptLen ) == 0 )
        {
            if( argv[i][nestedCascadeOpt.length()] == '=' )
                nestedCascadeName.assign( argv[i] + nestedCascadeOpt.length() + 1 );
            if( !nestedCascade.load( nestedCascadeName ) )
                cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                scale = 1;
            cout << " from which we read scale = " << scale << endl;
        }
        else if( tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0 )
        {
            tryflip = true;
            cout << " will try to flip image horizontally to detect assymetric objects\n";
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        /*capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;*/
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
        }
    }
    else
    {
        image = imread( "lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read lena.jpg" << endl;
    }

    cvNamedWindow( "result", 1 );

	int	seg_id;
	void * mem_ptr;
	int	semset_id;		/* id for semaphore set	*/
	Picture picShare;
	int pointNum = 0;
	unsigned char (*picFrame);
    if(1){
        cout << "In capture ..." << endl;

        for(;;)
        {
        	/* create a shared memory segment */
        	seg_id = shmget( TIME_MEM_KEY, SEG_SIZE, 0777 );
        	if ( seg_id == -1 )
        		oops("shmget",1);
        	/* attach to it and get a pointer to where it attaches */
        	mem_ptr = shmat( seg_id, NULL, 0 );
        	if ( mem_ptr == ( void *) -1 )
        		oops("shmat",2);
        	semset_id = semget( TIME_SEM_KEY, 2, 0);
        	void *mem_ptr_temp = mem_ptr;
        	wait_and_lock( semset_id );

    	    //frame.create(FRAME_HEIGHT,FRAME_WIDTH,CV_8UC3);
        	picFrame=new unsigned char[FRAME_HEIGHT*FRAME_WIDTH*3];
    	    for (int i = 0; i < 4; i++)
    	    {
    	    	memcpy(&pointNum,mem_ptr_temp,sizeof(int));
    	    	mem_ptr_temp = (int *)mem_ptr_temp + sizeof(int);
    	    	for (int j = 0; j < FRAME_HEIGHT*pointNum; j++)
    	        {
    	    		picFrame[j]=*(unsigned char*)mem_ptr_temp;
    	    		//printf("%d\t%u\n",j,*(unsigned char*)mem_ptr_temp);
    	            mem_ptr_temp = (unsigned char*)mem_ptr_temp + sizeof(unsigned char);
    	        }
    	    }
            Mat frame1(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, picFrame);
            frame = frame1;
        	release_lock( semset_id );
            for (int i=0; i<100; i++)
            {
            	cout << frame.row(i).col(i)<< endl;
            }
            cout << frame.row(1).total() << endl;
            cout << frame.col(1).total() << endl;
            cout << frame.row(1).col(1).row(0)<< endl;
            cout <<"type\t"<<frame.type() << endl;
            cout <<"depth\t"<<frame.depth() << endl;
            cout <<"channels\t"<<frame.channels()<<endl;
            cout <<"elemsize\t"<<frame.elemSize()<<endl;
            cout <<"elemsize1\t"<<frame.elemSize1()<<endl;
            if(frame.type()==CV_8UC3)
            {
                cout<<1000<<endl;
            }
            //return 0;
            if( frame.empty() )
                break;
            else
                frameCopy = frame;
            detectAndDraw( frameCopy, cascade, nestedCascade, scale, tryflip );
            delete picFrame;
            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }
        return 0;
        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    /*else
    {
        cout << "In image read" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            /*FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf), c;
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
                        c = waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }*/

    cvDestroyWindow("result");

    //return 0;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    int i = 0;
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    //cvtColor(gray,img,CV_GRAY2RGB);
    /*cout<<"img\t"<<img.row(1).col(1)<<endl;
    cout<<"gray\t"<<gray.row(1).col(1)<<endl;
    return;*/
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg(*r);
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            //|CV_HAAR_DO_CANNY_PRUNING
            |CV_HAAR_SCALE_IMAGE
            ,
            Size(30, 30) );
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    cout<<"img\t"<<img.row(1).col(1)<<endl;
    cout<<"gray\t"<<gray.row(1).col(1)<<endl;
    cv::imshow( "result", img );
}


