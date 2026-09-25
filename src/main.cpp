#include <Arduino.h>

const int MOUTH_PIN = 10;

/*
void setup(){
  pinMode(MOUTH_PIN, OUTPUT);
  digitalWrite(10,HIGH);
}
void loop(){

}
*/
void setup() {
  pinMode(MOUTH_PIN, OUTPUT);
  digitalWrite(MOUTH_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'O') {
      digitalWrite(MOUTH_PIN, HIGH);
    } else if (c == 'C') {
      digitalWrite(MOUTH_PIN, LOW);
    }
  }

}
  