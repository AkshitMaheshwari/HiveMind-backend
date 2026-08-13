#include<iostream>
#include<vector>
using namespace std;

int main(){
    vector<int> arr;
    int n;
    for(int i=0;i<=sqrt(n);i++){
        if(n%i==0){
            arr.push_back(i);

            //  dekh suppose 25 hain to 5*5
            //  ke baad vo same numbers repeat hi to honge
            //  1*25 5*5 fir wapis se 25*1 to sqrt ke 
            // badd multiple repeat hi hore
            //  to uske liye ye logic
            if(n/i!=i){
                arr.push_back(n/i);
            }
        }
    }
}