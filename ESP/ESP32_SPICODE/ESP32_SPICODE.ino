#include <ESP32SPISlave.h>
#include <WiFi.h>
#include <time.h>
#include <WString.h>

ESP32SPISlave slave;
WiFiServer WifiServer(12000);
WiFiClient client = WifiServer.available();

static constexpr uint32_t BUFFER_SIZE {12};
uint8_t spi_slave_tx_buf[BUFFER_SIZE];
uint8_t spi_slave_rx_buf[11];

// Replace with your network credentials
const char* ssid = "Upstairs";
const char* password = "123456789";

//Server we are sending the data to and connection timeout in ms
const char *ip = "13.41.53.180";
const uint port = 12000;
const int timeout = 3000;

//Time formatting using an NTP Server
const char* ntpServer = "0.uk.pool.ntp.org";
const long  gmtOffset_sec = 0;
const int   daylightOffset_sec = 0;
char currentTime[26] = {'\0'};

float bintofloat(unsigned int x) {
    union {
        unsigned int  x;
        float  f;
    } temp;
    temp.x = x;
    return temp.f;
}

//Initalise the Wifi Connection (can be re-ran for re-connection)
void initWiFi() {

  // Serial.println("[WiFi] | Checking the Current Connection");

  if(WL_CONNECTED != WiFi.status()){

    // Serial.println("--------------[WIFI COMM]----------");
    // Serial.println("[WiFi] | Disconnected From Wifi");
    WiFi.mode(WIFI_STA);
    // Serial.print("[WiFi] | SSID:");
    // Serial.print(ssid);
    // Serial.print(" PSWD:");
    // Serial.println(password);

    WiFi.begin(ssid, password);
    // Serial.println("[WiFi] | Starting Connection");
    
    while (WiFi.status() != WL_CONNECTED) {
      // Serial.println("[WiFi] | Waiting for Connection");
      delay(5000);
    }

    // Serial.print("[WiFi] | Connected at: ");
    // Serial.println(WiFi.localIP());

  }else {

    // Serial.println("[WiFi] | WiFi is Connected");

  }
  
}

bool printLocalTime(){

  initWiFi();

  struct tm timeinfo;
  if(!getLocalTime(&timeinfo)){
    Serial.println("[NTP] | Failed to obtain time");
    return false;
  }

  strftime(currentTime, 26, "%FT%TZ", &timeinfo);
  // Serial.print("[NTP] | ");
  // Serial.println(currentTime);
  return true;


}

void setup() {
  Serial.begin(115200);
  // Serial.println("====================[SETUP COMM]=================");
  //Connect to Wifi
  initWiFi();

  // Init and get the time
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  printLocalTime();

  //Start SPI Connection with HSPI Pins
  //HSPI = CS: 15, CLK: 14, MOSI: 13, MISO: 12
  slave.setDataMode(SPI_MODE0);
  slave.begin(HSPI);

  WifiServer.begin();

  slave.setQueueSize(64);
  // sendRequest("Cozzy");
}


void sendRequest(const char* Message) {
  // Serial.println("--------------[TPC COMM]-----------");

  initWiFi();

  //Check if connected to the server as a client
  if(!client.connected()){

      if(client.connect(ip,port,timeout)){

        // Serial.println("[TCP_IP]   | successful! server connected");
        
        //Send the Data if connected
        client.println(Message); 
        // Serial.print("[TCP_Tx] | ");
        // Serial.println(Message);

    } 
    else {

      // Serial.println("[TCP_IP]   | could not connect to server, timeout reached");
    
    }

  }

  else{
    // Serial.println("[TCP_IP]  | Open Connection Maintained");

    //Send the Data if connected
    client.println(Message); 
    // Serial.print("[TCP_Tx] | ");
    // Serial.println(Message);

    // while (!client.available());                // wait for response
    // // read entire response untill hit newline char
    // Serial.print("[TCP_Rx] | ");

    char val;
    // val = client.readStringUntil('\n');
    val = client.read();

    if(val != 255){

      // Serial.println(val);
      // Serial.println("--------------[SPI_Tx COMM]-----------");

      // //print the full Spi Tx Buffer
      // Serial.print("[SPI_Tx]  | ");
      // Serial.println(val);
      spi_slave_tx_buf[BUFFER_SIZE - 1] = val%48;
    }else{

      // Serial.println("No Server Response");
    }

  }

  // client.stop();

}


void loop() {

    bool successful_transfer = slave.wait(spi_slave_rx_buf, spi_slave_tx_buf, BUFFER_SIZE);

    // slave.yield();
    // if transaction has completed from master,
    // available() returns size of results of transaction,
    // and `spi_slave_rx_buf` is automatically updated
    while (slave.available()) {
        // Serial.println("=====================================[TRANSMISSION BLOCK]===================================");
        // Serial.println("--------------[SPI_Rx COMM]-----------");
        
        // //print the full Spi Rx Buffer
        // Serial.print("[SPI_Rx]  | ");
        // for(int i=0; i < slave.size(); i++){
        //   int val = spi_slave_rx_buf[i];
        //   printf("%d | ", val);
        // }
        // printf("\n");


        int check_bit = 0;
        for(int i=0; i<slave.size(); i++){
          check_bit |= spi_slave_rx_buf[i]; 
        }
        
        int step_count = (spi_slave_rx_buf[0] << 8) | spi_slave_rx_buf[1];
        uint pos_x = (spi_slave_rx_buf[2] << 24) | (spi_slave_rx_buf[3] << 16) | (spi_slave_rx_buf[4] << 8) | (spi_slave_rx_buf[5]);
        uint pos_y = (spi_slave_rx_buf[6] << 24) | (spi_slave_rx_buf[7] << 16) | (spi_slave_rx_buf[8] << 8) | (spi_slave_rx_buf[9]);
        float fx = bintofloat(pos_x);
        float fy = bintofloat(pos_y);
        int heading = spi_slave_rx_buf[10];

        //update the current time
        if(printLocalTime() && slave.size() > 0 && check_bit != 180){
          //buffer for the Json Data
          char PostData[256];

          // //Format the Json Data
          sprintf(PostData, "{\"timestamp\":\"%s\", \"device_id\":\"TestDevice#2\", \"total_steps\":%d, \"heading\":%d, \"pos_x\":%f, \"pos_y\":%f}", currentTime, step_count, heading, fx, fy);
          // Serial.print("[SPI->MSG] | ");
          // Serial.println(PostData);

          //send the formatted string
          sendRequest(PostData);

        }
        slave.pop();
    }
}
