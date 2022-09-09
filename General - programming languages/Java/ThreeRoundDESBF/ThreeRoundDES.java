import java.io.UnsupportedEncodingException;
import java.math.BigInteger;

public class ThreeRoundDES {

	public static String DES(String plaintext, String key)
	{
		String[] subkeys = createSubkeys(key);
		
		String[] L = new String[4];
		String[] R = new String[4];
		L[0] = seperateLeftRight(plaintext,"LEFT"); 
		R[0] = seperateLeftRight(plaintext,"RIGHT"); 
		
		for(int i=0; i<2; i++)
		{
			L[i+1] = R[i];
			R[i+1] = XOR(L[i],f_func(R[i],subkeys[i]));
		}
			// to avoid intersection
			L[3] = XOR(L[2],f_func(R[2],subkeys[2]));
			R[3] = R[2];
		
		String cypher = L[3] + R[3];	
		String hex = binaryToHex(cypher);

		return hex;
	}
	public static String hexToBin(String s) 
    {
		return new BigInteger(s, 16).toString(2);
    }
	
	public static String[] createSubkeys(String key)
	{
		String[] subkeys = iterations(splitKeyIntoLeftRight(pc1Table(key)));
		for(int i=0; i<3; i++)
		{
			subkeys[i] = pc2Table(subkeys[i]);
		}
		return subkeys;
	}
	
	public static String stringToBinary(String word)
	{
		return catStringsIntoOne(stringToBinaryArray(word));
	}
	
	public static String[] stringToBinaryArray(String word)
	{
		String[] binaryNum = new String[word.length()];
		for(int i=0; i<word.length(); i++)
		{
			binaryNum[i] = (Integer.toBinaryString((int)word.charAt(i)));
			while(binaryNum[i].length()<8){
				binaryNum[i] = "0" + binaryNum[i];
			};
		}
		return binaryNum;
	}

	public static String catStringsIntoOne(String[] arr)
	{
		String str = "";
		for(int i=0;i<arr.length;i++){
			str += arr[i];
		}
		return str;
	}


	
	public static String[] splitKeyIntoLeftRight(String key)
	{
		String[] arr = new String[2];
			arr[0] = key.substring(0, 28);
			arr[1] = key.substring(28, 56); 
		return arr;
	}

	public static String[] iterations(String[] keys)
	{
		String [][] newkeys = new String[2][3];
		int[] numberLeftShifts = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};
		String[] subkeys = new String[3];
		for(int i=0;i<3;i++){
			for(int j=0;j<numberLeftShifts[i];j++){
				keys[0] = shiftLeft(keys[0]);
				keys[1] = shiftLeft(keys[1]);
			}
			newkeys[0][i] = keys[0];
			newkeys[1][i] = keys[1];
			subkeys[i] = newkeys[0][i] + newkeys[1][i];
		}
		return subkeys;
	}
	
	public static String shiftLeft(String str)
	{
		str = str.substring(1, str.length()) + str.charAt(0);
		return str;
	}
	
	public static String seperateLeftRight(String plaintext, String side)
	{
		if(side == "LEFT")
			return plaintext.substring(0, 32);
		else //if(side == "RIGHT")
			return plaintext.substring(32, 64);
	}

	public static String XOR(String left, String right){
		String XOR = "";
		for(int i=0; i<left.length(); i++){
			if((left.charAt(i) == '0' && right.charAt(i) == '0') || (left.charAt(i) == '1' && right.charAt(i) == '1'))
				XOR += "0";			
			else
				XOR += "1";
		}
		return XOR;
	}
	public static String f_func(String right, String key)
	{
		right = E(right);
		String res = P(SBox_func(XOR(right, key)));
		return res;
	}
	
	public static String binaryToHex(String input)
	{
		String[] temp = new String[8];
		String key = "";
		for(int i = 0; i < 8 ; i++){
			temp[i] = input.substring(i*8,(i+1)*8);
			key += Integer.toHexString(Integer.parseInt(temp[i],2));
		}
		return key;
	}
	
	public static String Convert6BitsInto4(String _6bit, int[][] sBox)
	{
		String row = "";
		String col = "";
		row += String.valueOf(_6bit.charAt(0)) + String.valueOf(_6bit.charAt(5));
		col += String.valueOf(_6bit.charAt(1)) + String.valueOf(_6bit.charAt(2))+ String.valueOf(_6bit.charAt(3)) + String.valueOf(_6bit.charAt(4));
		int bin = sBox[Integer.parseInt(row,2)][Integer.parseInt(col,2)];
		String res = Integer.toBinaryString(bin);
		if(res.length()<4){
			do{
				res = "0" + res;
			}while(res.length()<4);
		}
		return res;
	}
	
	public static String SBox_func(String input)
	{
		int[][] s1 = {
				{14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0,  7},
				{0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8},
				{4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0},
				{15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13}
		};
		int[][] s2 = {
				{15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10},
				{3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11,5},
				{0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15},
				{13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9}
		};
		int[][] s3 = {
				{10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8},
				{13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1},
				{13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7},
				{1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12}
		};
		int[][] s4 = {
				{7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15},
				{13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9},
				{10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4},
				{3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14}
		};
		int[][] s5 = {
				{2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9},
				{14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6},
				{4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14},
				{11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3}
		};
		int[][] s6 = {
				{12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11},
				{10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8},
				{9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6},
				{4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13}
		};
		int[][] s7 = {
				{4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1},
				{13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6},
				{1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2},
				{6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12}
		};
		int[][] s8 = {
				{13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7},
				{1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2},
				{7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8},
				{2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11}
		};

		String[] _6bit = new String[8];
		for(int i=0;i<8;i++)			
			_6bit[i] = input.substring(i*6, (i+1)*6);
		
		String[] _4bit = new String[8]; 

		_4bit[0] = Convert6BitsInto4(_6bit[0],s1);
		_4bit[1] = Convert6BitsInto4(_6bit[1],s2);
		_4bit[2] = Convert6BitsInto4(_6bit[2],s3);
		_4bit[3] = Convert6BitsInto4(_6bit[3],s4);
		_4bit[4] = Convert6BitsInto4(_6bit[4],s5);
		_4bit[5] = Convert6BitsInto4(_6bit[5],s6);
		_4bit[6] = Convert6BitsInto4(_6bit[6],s7);
		_4bit[7] = Convert6BitsInto4(_6bit[7],s8);
		String res = _4bit[0] + _4bit[1] + _4bit[2] + _4bit[3] + _4bit[4] + _4bit[5] + _4bit[6] + _4bit[7];
		return res;
	}
	
	public static String pc1Table(String input)
	{
		int[] pc1 = 
				{
				57, 49, 41, 33, 25, 17, 9,
				1,58, 50, 42, 34, 26, 18,
				10, 2, 59, 51, 43, 35, 27,
				19, 11, 3, 60, 52, 44, 36,
				63, 55, 47, 39, 31, 23, 15,
				7, 62, 54, 46, 38, 30, 22,
				14, 6, 61, 53, 45, 37, 29,
				21, 13, 5, 28, 20, 12, 4
				};
		String res = "";
		for(int i=0;i<pc1.length;i++){
			res += input.charAt(pc1[i]-1);
		}		
		return res;
	}
	
	public static String pc2Table(String input)
	{
		int[] pc2 = {
				14, 17, 11, 24, 1, 5,
				3, 28, 15, 6, 21, 10,
				23, 19, 12, 4, 26, 8,
				16, 7, 27, 20, 13, 2,
				41, 52, 31, 37, 47, 55,
				30, 40, 51, 45, 33, 48,
				44, 49, 39, 56, 34, 53,
				46, 42, 50, 36, 29, 32
		};
		String res = "";
		for(int i=0;i<pc2.length;i++){
			res += input.charAt(pc2[i]-1);
		}		
		return res;
	}
	
	public static String E(String right)
	{
		int[] Etable = {
				32, 1, 2, 3, 4, 5,
				4, 5, 6, 7, 8, 9,
				8, 9, 10, 11, 12, 13,
				12, 13, 14, 15, 16, 17,
				16, 17, 18, 19, 20, 21,
				20, 21, 22, 23, 24, 25,
				24, 25, 26, 27, 28, 29,
				28, 29, 30, 31, 32, 1
		};
		String res = "";
		for(int i=0;i<Etable.length;i++)
			res += right.charAt(Etable[i]-1);
		
		return res;
	}
	
	public static String P(String input)
	{
		int[] ptable = {
				16, 7, 20, 21,
				29, 12, 28, 17,
				1, 15, 23, 26,
				5, 18, 31, 10,
				2, 8, 24, 14,
				32, 27, 3, 9,
				19, 13, 30, 6,
				22, 11, 4, 25
				};
		String res = "";
		for(int i=0;i<ptable.length;i++)
			res += input.charAt(ptable[i]-1);
		
		return res;
	}
	
	public static String bruteForce(String plaintext,String cypher)
	{
		byte[] key=new byte[8];
		key[0]=0b0;
		key[1]=0b01000001;
		
		int[] index={0,0,0,0,0,0,0};
		
		// start brute force loop
		for(index[0]=0; index[0]<52; index[0]++) //8-16
		{
			for(index[1]=0, key[2]=0b01000001;index[1]<52;index[1]++) //16-24
			{
				for(index[2]=0, key[3]=0b01000001;index[2]<47;index[2]++) //24-32
				{
					if(index[2]==15)//bit 28 is 0
					{
						index[2]+=16;//jump +16 +1(after continue)=+17 to jump right to next a(after O)
						key[3]+=17;//must increment the key too from O to a.
						continue;
					}
					
					for(index[3]=0, key[4]=0b01000001;index[3]<52;index[3]++) //32-40
					{
						for(index[4]=0, key[5]=0b01000001;index[4]<52;index[4]++) //40-48
						{
						
							for( index[5]=0, key[6]=0b01000001;index[5]<52;index[5]++) //48-56
							{
							
								for(index[6]=0,key[7]=0b01000001;index[6]<52;index[6]++) //56-64
								{
									try {
										String newKey=new String(key,"ASCII");
										System.out.println(newKey);
										
										if(ThreeRoundDES.DES(plaintext, stringToBinary(newKey)).equals(cypher))
											return newKey;

									} catch (UnsupportedEncodingException e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
									}
									if(index[6]==25)
										key[7]+=6;
							
									key[7]++;	
								}
								
								if(index[5]==25)
									key[6]+=6;
								
								key[6]++;
							}	
							if(index[4]==25)
									key[5]+=6;
							
							key[5]++;	
						}
						if(index[3]==25)
							key[4]+=6;
						
						key[4]++;				
					}	
					if(index[2]==25)
						key[3]+=6;
					
					key[3]++;
				}				
				if(index[1]==25)
					key[2]+=6;
				
				key[2]++;
			}	
			if(index[0]==25)
				key[1]+=6;
			
			key[1]++;
		}		
		
		return "Not found";
	}
		
	public static void main(String[] args) 
	{
		String key=bruteForce(catStringsIntoOne(stringToBinaryArray("nonsense")),"d8164228f290cbaf");
		System.out.println(key);
	}
	
}
