#include <iostream>
#include <string>
#include <cmath>
using namespace std;
#include <map>
map<string, int> calcLetterFrequency(const string& text)
{
	/*
	@params string text
	creates the map of letter to it's frequency in the text
	@returns string (letter in english) to float(frequency in text)
	as map object
	*/
	map<string, int> freq;
	for(char c='A';c<='Z';c++)
	{
		string init(1, c);
		freq[init]=0;

	}
	for(int i=0;i<text.size();i++)
	{
		char c=text[i];
		string key(1, c);
		freq[key]++;
		
	}
	return freq;
}
string createKey(const string& plainText,const string& keyWord)
{
	/*
	@params string plainText, string keyWord
	repeating the keyWord according to the length of the plainText
	to create the key to encrpyt/decrypt Vigenere plain/cipher text.
	@retuns string key , the key to Vigenere algo.
	*/
	int length = plainText.size();
	string key="";
	for (int i = 0;length>0;length--)
	{
		if(i == keyWord.size())
			i = 0;
		key.push_back(keyWord[i]);
		i++;
	}

	return key;

}

string Encryption(const string& plainText, const string& key)
{
	/*
	@params string plainText, string key
	ecrypting the plainText via the key in the Vigenere algo
	@returns string cipherText
	*/
	string cipherText;
	string vigenereKey=createKey(plainText,key);
	for (int i = 0; i < plainText.size(); i++)
	{
		// converting in range 0-25
		int encryped_letter = (plainText[i] + vigenereKey[i]) %26;
		// convert into alphabets(ASCII)
		encryped_letter += 'A';
		cipherText.push_back(encryped_letter);
	}
	return cipherText;
}

string Decryption(const string& cipherText,const string& key)
{
	/*
	@params string cipherText, string key
	decrypting the cipherText via the key in the Vigenere algo
	@returns string decrypted text
	*/
    string plainText;
    string vigenereKey=createKey(cipherText,key);
    for (int i = 0 ; i < cipherText.size(); i++)
    {
        // converting in range 0-25
        int decryped_letter = (cipherText[i] - vigenereKey[i] + 26)%26;
 
        // convert into alphabets(ASCII)
        decryped_letter += 'A';
        plainText.push_back(decryped_letter);
    }
    return plainText;
}
float indexOfCoincidence(const string& cipherPart)
{
	/*
	@params string cipherPart 
	calculating the IC of this part of the cipherText
	@returns float index of coincidence, the IC of this part
	*/
	float numerator=0;
	float denominator=cipherPart.size() * (cipherPart.size()-1);
	map<string, int> freq=calcLetterFrequency(cipherPart);
	for(char c='A';c<='Z';c++)
	{
		string init(1, c);
		numerator+=freq[init]*(freq[init]-1);

	}
	return 26*(numerator/denominator);
}
int calcIC(const string& cipherText)
{
	/*
	@params cihper text
	predicing the key length with the highest probability using the 
	help of the function indexOfCoincidence
	@returns int length, the predicted length of the key
	*/
	//first calc for key length 1 because there is no need to divide it to pieces
	float max_avg_IC=indexOfCoincidence(cipherText);
	int predictedLength=1;
	for(int dist=2;dist<=15;dist++)
	{
		float avg_IC=0.0;
		float sum_IC=0.0;
		string * seqs=new string[dist];
		for(int i=0,j=0;i<cipherText.size();i++,j++)
		{
			if(j==dist)
			{
				j=0;
			}
			seqs[j].push_back(cipherText[i]);
			
		}
		for(int i=0;i<dist;i++)
		{
			sum_IC+=indexOfCoincidence(seqs[i]);
		}
		avg_IC=sum_IC/dist;
		if(abs(1.73-max_avg_IC)>abs(1.73-avg_IC))//1.73 is for english,the close the better
		{
			max_avg_IC=avg_IC;
			predictedLength=dist;
		}
		delete [] seqs;
	}
	return predictedLength;
}
map<string,float> getEnglishLetterFrequency()
{
	/*
	creates the map of letter in english to it's known frequency
	according to wikipedia
	@returns string (letter in english) to float(known frequency in english)
	as map object
	*/
	map<string,float> englishLetterFreq;
	englishLetterFreq["A"]=0.08167;
	englishLetterFreq["B"]=0.01492;
	englishLetterFreq["C"]=0.02782;
	englishLetterFreq["D"]=0.04253;
	englishLetterFreq["E"]=0.12702;
	englishLetterFreq["F"]=0.02228;
	englishLetterFreq["G"]=0.02015;
	englishLetterFreq["H"]=0.06094;
	englishLetterFreq["I"]=0.06966;
	englishLetterFreq["J"]=0.00153;
	englishLetterFreq["K"]=0.00772;
	englishLetterFreq["L"]=0.04025;
	englishLetterFreq["M"]=0.02406;
	englishLetterFreq["N"]=0.06749;
	englishLetterFreq["O"]=0.07507;
	englishLetterFreq["P"]=0.01929;
	englishLetterFreq["Q"]=0.00095;
	englishLetterFreq["R"]=0.05987;
	englishLetterFreq["S"]=0.06327;
	englishLetterFreq["T"]=0.09056;
	englishLetterFreq["U"]=0.02758;
	englishLetterFreq["V"]=0.00978;
	englishLetterFreq["W"]=0.02360;
	englishLetterFreq["X"]=0.00150;
	englishLetterFreq["Y"]=0.01974;
	englishLetterFreq["Z"]=0.00074;
	return englishLetterFreq;
}
char getPredictedLetter(const string& column)
{
	/*
	@params the column of the cipherText 
	after the cipher text is divided into keyLength columns, this function
	takes the column and try to reveal the #i letter of the key
	via the #i column , when using the greatest corelation sum(f*n)
	@returns char letter, which is the predicted #i key letter 
	*/
	map<string,float> englishFreq=getEnglishLetterFrequency();
	char predictedChar='*';
	float max_cor=0;
	for(char c='A';c<='Z';c++)
	{
		string decryped="";
		for(int i=0;i<column.size();i++)
		{
			int decryped_letter = (column[i] - c + 26)%26;
        	// convert into alphabets(ASCII)
        	decryped_letter += 'A';
        	decryped.push_back(decryped_letter);
		}
		map<string, int> decryped_freq=calcLetterFrequency(decryped);
		float cor=0;
		for(char ch='A';ch<='Z';ch++)
		{
			string str(1,ch);
			cor+=(float(decryped_freq[str])/float(26))*englishFreq[str];
		}
		
		if(cor>max_cor)
		{
			max_cor=cor;
			predictedChar=c;
		}
	}
	
	return predictedChar;
}

string revealKey(const string& cipherText,const int keyLength)
{
	/*
		@params the ciphertext and key length
		the fuction is trying to reveal the key via the cihper key and key length
		using the helper function getPredictedLetter
		@returns string key, which is the predicted key 
	*/
	int cipherSize=cipherText.size();
	int len=(int)cipherSize/keyLength;
	bool notFullBlock=false;
	if(cipherSize%keyLength>0)//if there is a need to another string block
	{	
		len++;
		notFullBlock=true;
	}
	string * seqs=new string[len];
	for(int i=0,j=0;i<cipherSize;i+=keyLength,j++)
	{
		if(cipherSize-i<keyLength && cipherSize-i>0)
			seqs[j]+=cipherText.substr(i,cipherSize-i);
		else
			seqs[j]+=cipherText.substr(i,keyLength);
	}
	string * cols=new string[keyLength];
	for(int i=0;i<keyLength;i++)
	{
		for(int j=0;j<len-1;j++)
		{
			cols[i].push_back(seqs[j][i]);

		}
		if (notFullBlock)//if we are in the last sequence,must check if it full block or no, to avoid outOfIndex
		{
			if(seqs[len-1].size()==i)
				continue;
		}
		else
		{
			cols[i].push_back(seqs[len-1][i]);
		}
		
	}
	delete [] seqs;
	string predictedKey="";
	for(int i=0;i<keyLength;i++)
	{
		predictedKey.push_back(getPredictedLetter(cols[i]));
	}
	delete [] cols;
	return predictedKey;
}
int main()
{
	
	string cipherText="HUGVUKSATTMUNDKUMKVVAYVLPOMCEDTBGKIIEYARTREEDRINKFSMEMQNGFEHUVMAMHRUCPVVHBWMOGYZXVJWOMKBMAIELJVRPOMCEDRBWKIUNZEEEFRRPKMAZZYUDZRYRALVRZGNFLEKAKTVGNEJOAWBFLSEEBIAMSCIAKTVGNVRPKMAZHXDYXLNFIIIDJSEMPWJOHIIBZMKOMMZNAXVRZHGTWTZNBEGFFGYAHFRKKSFRJRYRALZSVRQGVXYIIKZHYIRHYMFMPRTTGCVKLQVMWIEBAARSDRGALFCEVOQXJIDBZVNGKIRCCWRIHVRTZHLBUKVMWIEPYSLGCXVMZKYONXHIVRKHZJYHVVVABIEEFMNINLRWALVMJVEHDZRIIPLBOEUSJYTAAXFBJVEHDJIOHQLUVSBSNYEVLEJEJJFNYVFWNSEKVAWOMXUXSSJTGIAHYIWOMXUXYEIEVRQKHHZAIXZTPHVNRLBFALVAIKREZRRMZPRGVVVNVQRELWJHZVRYVVVVZVZHYIRNYXUXZMCKZRFTKYECZVGTPRIUNXYBUKFFZEPAWYIPGIPNYXRIIXUKPPCEYQRYPPCEYQRPPXYFVRGTZXZCOIEKVVJNZZRKMICTWISHYIJOOLNMUSNTJWGBSPKHZFRTAMEGJJZROIRROMFMVSURZTRTAMEGOMFLVQVVDWVMVVVNOVRTAMEGZRGKHRTEVXZRJLRMWIEWVSISJQREHXVVDWVMVVVNOVRTAMEGZRGKHRTEVXZRJLRMWIEWVSITCMFBZMKAIHAHALZNBQBKLTIENIAMSCDYNSHENVVWNXEHUKVRCIFBAEKIIKGALREOGSAZLVJIMWNBKMFRHEQTTXIUGCLHBVWOMKVOLRVSNMVFWPFRZFHMALVFVGGBZMNANRNIWMEGVRQLVKVNOPLRVYTAHIETWTZNBEAWZSWADRGEFCFUXEZXAEGPDRTMHTGIIKNMTCTHVQOXYHFOMXUTAMJCVVPXDEJSPVRBOIRRYCBNOIIEDSCXUIUWDHRMOIUOJVQTYOEENWGALVVAIHAHALZNBQBKLHVEKMAMVXYEYEEDUIJSKIRKPRXLJRTBZXFOYXUXYINOIHRKPRXFZEEBUKUOPFGBUKURZEZBUKURZEZLUSDOMXNEZIMEMHNKLHKOYVRTTFVFJVRUBXKHZWVELRTEREFNUFIOFIATUHKHZWGBSPEENWTTCIEOOSXXUEEDOLRHUPPWJVQMOIIENTBDLRNANXUXDLZSKIEXKAFRYPRGVVVTCMFBDLZSKIEXKEEDVRRVOSDUMQHKLHSAXOGALAFRYPRGVVVMZVREFXYINEAWUSKHDRTFVVVBVGXBUXFTCIPAHQSEMXHKUMEGVPYFFWFUGAVMOMEMZFHKUMEGNSBGHKRIIMUXHVUAOECIPRXSJQRMOMEGGSHWLVKHVROXMSIENYEXSCJADHVLBVVLTXUTAMJSJQRMOMEGVXZRDMEDJAYTAXZCZPRMTIJEZXUXUAYAOXUXYIRTDWNGKXYINQLLAIIYZBCEVVVLZXZROIRROFRLAMCLVQBFLRKAIHGAPWDYNXRKFIOPGSEXAMJTCIJBUHRNYRBMOMEGHSEXVTVNCIEXPJCUIKGALWYUOXRKDLVNRMGATEEYVJYBYXRNYJYNAXVRDRGALVVSOICILHRSOEGXSCIAQIAHMXYENEVGAPPDVCFHMCFRZRBMALVLZEFMVFVINEAVLQRDZLRGVXRMDRHMLWKOKTRWVVJTVCRWOISUOAVMOQZEISSEVVUOMPNWFTVRXLRWHFFVZQLVOEDBZVQHVVGEMGUXKYGOIEONZXFFKEYEHWAUNXNUVZVMTGUTTFVRYSBKWIICCIQTUHJAOEAWUSKHDRTFVVVTCIAMOMJEWSARIMIDWITNPPZNBQLLHHWAIGLBUXFSHMYBUKSYOLRZYEMEVRQLAIINYIPHYYDOAXUXJSLNOIATUGVIOABKLXYOPKUMOCTRZWGULWYOMRNGKWYAQIAMOSLINEVWHVKSPVRGVGIAQIAZOEJTGCTKPQRNYEAVPIETMEIXUARNYIEBUKWRJQGALRZGCXYRZLFRZXRESQVWCEGMOICOMHYRUEDEDWBGALVNDKUMZTCUOSABHRJHJVRJBSKHOLRKHZVNIIIXYQFRZQHVOMDAMZRESIUTCMFNUKRIIPLYVACTJLRTYHZSXSHKZIJOKPNBUPPTCSHZOMKSVRFPLVCIOXYXTIRNDRTEPXKLZVRELZRNXCOHYIWOMARVHREOOLREWEXRZIVGNXYAORBEPZZNBLHFHRSEDRTXCIIYZXJTZFCENWRWDMKHNIRBUKSIMHNUVZVHDWPAHQSEMHBHYFZRYSEULEJTPTBGALVSXYYIAYIEYFHLAESOQIUBZGYAHFRKKSFRRMGAZYTHIEZXHWEEQIEFVVVBPXGALVRVZRFBAXZNBPBGLPPOIXUTATCAXMQUBWKSKSXXVRCYOLNMVRVWJVQTZMWHDWFHBPZNOLNMVRVWJVQALHZDJYGIVYINJXUBUKWUMXUXYXYEILRNAXVRZHAHAEWEVXUXYXYEILRYSYKTZVRWAMCLDWPTYGVLTQBKLXYAIQHMAIIEYSGALVWRDIAWZLRVZJYHDRSEASEXVRKHZQBKYSNHZAVESPVAQIZXHWDYCSCXZLRVZJYHDRSEASEXALVNOLRUPVUSVMQGLZVRHSEXZXRROPRWHXKHZWGBSPEENWOKVOVNWCEXWPPSJECMSCJPJORGKSLBOPRLZWRIYMJAHXZTPXGXYWZSDXFHUPPSOSPDHRUSOSEXJELGCXSKVQJOHIHGOEGPTQNLAIIWCSZNUQVRXMSNSHZSVWGXYJFLGSJXKJRSOEAWMSCLJARWMEJTZVGBSPYINWBGNWFNZFHKKIEBJVRMPPCTCIQBYKVSJJUBZLFPZXUTAQVLVRPAVPPBPVQXUFFRZSSGLZVRIIIXYQFRZFHMALVRVZRGZXZLGFRZBMCIIKNESQPFVRPRPRKONQVEPRXSOVNBNLKIRLRXSIUAXYFAPSEEYWRTAMEFMSAMVJSIMHNGKFLSOEAWKSFROLRGBTFNOLROLPMEOWVGRMEGDFRMVSBMTWREMXFLDRXBUKWAIGLNUXFFVRPRALZNFMAZDLRTOLVLVQZNJYFUPVUOACBKLAYAOXUBZKIIHYAZHMELTKUTZXCYBEHGAEEDJQVGVYJBDVQHMCFRZQRTUXZNXVBTRMEGIIIXYQFRZXUNZMJAOIAZHKVDDRTNLWJIIKONARFSTPYTIPVESTEXZWZNBXBMOIWORPJAVWVFDIERLCVSISJUBVEEYMAMVQPBJWBFZGFRZXUBZEEDHSEXPWRTYMIBUMEGRMGATCYEVHNMLEJEMIPEPRZNBSAMOITUNLVHUWMEGZRMSMEIIKGAHXKHZPNFWPZGCXTEVEKEYSRKIYKWCSFXCICVZXIBVPVTGMABUKNIOLGALPRMKPVZOXXLJEGBUKFEMWUXZLRLGTEXZWRHIIIXYQFRZXUXUQVTCSHZOXKHZEVKNVVWYIALLVGEMJHFLHWRJQNGBRJEZRPXUWVRNAHGNFPSZVNIOMDWCSFXMSFTAEYEZXZNFPRWVRKHZXHYAIUFGSBKDVVTXLVVYMVDOLLZVHYAOLYXUXKHZIORALVSZEAZLPJHZLNMOWVNOXUXLVVSKMGXYIJPDXRTUHEEKIAMOIWRJQGAFQVMJVVXZSWLZRBKLULAAJBJBEWFOLVLRMEDIICXUXYEVRQYVVXEOXUBZPFSOPRGVVVQPSGAALVRVZRGUIMEMQBKLTIOKLRMZEZDDXUBUKFFZZVEWVFPCIGLAMCLDJOBYHFRYIIBSAYEOLRKAIDPOIELLRKOMAUXALVROIZILWKTJWFXKXYEZLRKLEJHJVRWLWFLVXRRLXRLGYAWHYETZHBGALZSYIFXYXCAIHRGJLRNOIQHUXYINLBFLFPHJVEHYLRUIXRWAICLHIGKBPPIDQCEVVVINXUXYIZSOLRKLFRLHMAZPPVAYXRESQVTZPYFLMZMKPBKLULOOLGALVRVZRAXCIIMJVRIYSGHZXFTPHZTCMAZVJVVDPCKVTYEOWGBSPZFWMEWVVUEQMYUFXYAOLRTCIETCEGULRUSVFBOLYJBTXUTAKFDRIOHALRDJVRMLPCTCMFLVYCWDXULVVIORPNWLRZFRMGAPRKHZHVLAEETVMQXURZTNLNESGCANTNLHMETZHZTPHVNRLBFALVAIKREZRRMZPRGVVVCGEFIHVRRZEAWYEUIVRGFHMUEIAUHTXYEVRTXSWEAHIYXUSIELYBMOXYEMEIXURVVZVZHYISEOLNMDSIDJYELPKEOATNKAMEGWMEWVVWIZRQBZLIIZORWBTJTVVGBUKXEOXUXLFRCFMAMVXYEOIZILWKAIHGALRZGCXFISYKOIMNGZLFRZPRTCIEOWPNVRTCUHINLHXFKZRBYALRTGMRMOCJOPPUTALJPJORGSIRVZQLEVRVLDRRLZYEBMSXXUULIOXUXIYJTVFBOLQPDJSEMHOVTCCOXHOWRJQBNAQPHZEEMHRUTVORMOCWOMQSKVQFFAQLWVSIQPSGAALVRVZRGUIMEMQBKLEEDOLRKHZVNIIIXYJCIOXVGNWKIGPVLZMKTDRTLAMCLDWFBAXZNBSAMOIGAGPVWIYJTJJCTSPRSEYFMHFFVZQLVOEDBZVQHVVRNYLVLLCVSCEIXHPCTCIFXLQZNBSSTKIDOIWGAHXZSYVRTTMEGVRQMOICAHTYBNLKOZVUBTWKRZEZBUKKHMSJLALVSCEQHDSETCISEVSIAIHZRZSLLAVBFVYKTCEGLOEUORXUTAPZENJYHHXZNBSAMOIWLJSELOECLWIYBMXVDIIIXYQFRZ";
	int predictedLength=calcIC(cipherText);
	cout << "predicted length of the key is:"<<predictedLength<<endl;
	string predictedKey=revealKey(cipherText,predictedLength);
	cout << "predicted key:" <<predictedKey << endl;
	cout << "decryped text:"<<endl << Decryption(cipherText,predictedKey)<<endl;
	return 0;
}
