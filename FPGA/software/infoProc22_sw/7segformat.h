#include <sys/alt_stdio.h>
#include <stdio.h>
#include "altera_up_avalon_accelerometer_spi_regs.h"
#include "altera_avalon_timer_regs.h"
#include "system.h"
#define CHARLIM 128	//Maximum character lengthofText of what the user places in memory.  Increase to allow longer sequences
int TrackPosition = 0,lengthofText;
char enteredText[CHARLIM];

//Takes the user's input and only uses the allowed letters.  Returns the lengthofText of the string entered
void getActualText(char text[CHARLIM]){
	int idx = 0;	//We need two indicies because the entered and actual text sequences need not be aligned
	char currentLetter; //Keeps track of the character we are wanting to add
	//Go through each letter in the entered text
	for (int i = 0; i < lengthofText; i++){
		currentLetter = text[i];
		if (currentLetter > 96){
			//Gets only the uppercase letter
			currentLetter -= 32;
		}
		switch(currentLetter){
		case 'M':
			//We build the letter "M" from two "n's," so we need to change the index twice in the actual text
			enteredText[idx] = 'N';
			enteredText[idx + 1] = 'N';
			idx += 2;
			break;
		case 'W':
			//We build the letter "W" from two "v's," so we need to change the index twice in the actual text
			enteredText[idx] = 'V';
			enteredText[idx + 1] = 'V';
			idx += 2;
			break;
		default:
			//Copy the new letter into the actual text
			enteredText[idx] = currentLetter;
			idx++;
		}
	}
	lengthofText =idx;
}
//This function clears the text on the display:
void findL(char text[CHARLIM]){
	int i=0;
	while(i < CHARLIM && text[i] != '\0'){
		i++;
	}
	lengthofText = i;
}

//Gets the binary representation of the character
int getBin(char letter){
	/*Based on the character entered, we convert to binary so the 7-segment knows which lights to turn on.
	The 7-segment has inverted logic so a 0 means the light is on and a 1 means the light is off.
	The rightmost bit starts the index at HEX#[0], and the leftmost bit is HEX#[6], the pattern
	for the 7-segment is shown in the DE0_C5 User Manual*/
	switch(letter){
	case '0':
		return 0b1000000;
	case '1':
		return 0b1111001;
	case '2':
		return 0b0100100;
	case '3':
		return 0b0110000;
	case '4':
		return 0b0011001;
	case '5':
	case '6':
		return 0b0000010;
	case '7':
		return 0b1111000;
	case '8':
		return 0b0000000;
	case '9':
		return 0b0010000;
	case 'A':
		return 0b0001000;
	case 'B'://Lowercase
		return 0b0000011;
	case 'C':
		return 0b1000110;
	case 'D'://Lowercase
		return 0b0100001;
	case 'E':
		return 0b0000110;
	case 'F':
		return 0b0001110;
	case 'G':
		return 0b0010000;
	case 'H':
		return 0b0001001;
	case 'I':
		return 0b1111001;
	case 'J':
		return 0b1110001;
	case 'K':
		return 0b0001010;
	case 'L':
		return 0b1000111;
	case 'N':
		return 0b0101011;
	case 'O':
		return 0b1000000;
	case 'P':
		return 0b0001100;
	case 'Q':
		return 0b0011000;
	case 'R'://Lowercase
		return 0b0101111;
	case 'S':
		return 0b0010010;
	case 'T':
		return 0b0000111;
	case 'U':
		return 0b1000001;
	case 'V':
		return 0b1100011;
	case 'X':
		return 0b0011011;
	case 'Y':
		return 0b0010001;
	case 'Z':
		return 0b0100100;
	default:
		return 0b1111111;
	}
}


void hex_write(char text[CHARLIM]){
	TrackPosition=0;
	// for(int i=0;i <CHARLIM;i++){
	// 	enteredText[i] = text[i];
	// }
	findL(text);
	getActualText(text);
	if(lengthofText<=6){
		for(int i = lengthofText;i<6;i++){
			enteredText[i] = '\0';
		}
	}
	IOWR(HEX5_BASE, 0, getBin(enteredText[0]));
	IOWR(HEX4_BASE, 0, getBin(enteredText[1]));
	IOWR(HEX3_BASE, 0, getBin(enteredText[2]));
	IOWR(HEX2_BASE, 0, getBin(enteredText[3]));
	IOWR(HEX1_BASE, 0, getBin(enteredText[4]));
	IOWR(HEX0_BASE, 0, getBin(enteredText[5]));
}

void shift7seg(){
	TrackPosition++;
	if(lengthofText <= 6) return;
	IOWR(HEX5_BASE, 0, getBin(enteredText[(TrackPosition)%lengthofText]));
	IOWR(HEX4_BASE, 0, getBin(enteredText[(TrackPosition+1)%lengthofText]));
	IOWR(HEX3_BASE, 0, getBin(enteredText[(TrackPosition+2)%lengthofText]));
	IOWR(HEX2_BASE, 0, getBin(enteredText[(TrackPosition+3)%lengthofText]));
	IOWR(HEX1_BASE, 0, getBin(enteredText[(TrackPosition+4)%lengthofText]));
	IOWR(HEX0_BASE, 0, getBin(enteredText[(TrackPosition+5)%lengthofText]));
}