#include <iostream>
#include <cstdlib>
#include <cstdio>
using namespace std;
int main(int argc,char* argv[]){
    if(argc!=2){
        cerr<<"Usage: "<<argv[0]<<"N"<<endl;
    }
    int N=atoi(argv[1]);
    for(int i=0;i<=N;i++){
        printf("%d",i);
        if(i<N) printf(" ");
    }
    printf("\n");
    for(int i=N;i>=0;--i){
        cout<<i;
        if(i>0) cout<<" ";
    }
    cout<<endl;
    return 0;
}