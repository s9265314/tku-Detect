int buttonState;
int buttonPin;
int ledPin;
int i;
int count = 0;
String str;
#include <Servo.h>
Servo myservo;
void setup() {
  myservo.attach(9);
  Serial.begin(9600);
  myservo.write(0);
}

void loop() { 

  if (Serial.available()) {
    // 讀取傳入的字串直到"\n"結尾
    str = Serial.readStringUntil('\n');
    
    if (str == "on" && count == 0) {
      count++;
      for(int i = 0; i <= 60; i+=1){
          myservo.write(i); // 使用write，傳入角度，從0度轉到180度
          delay(20);
          str="off";
          myservo.write(0);
       }
    }
 }
}
