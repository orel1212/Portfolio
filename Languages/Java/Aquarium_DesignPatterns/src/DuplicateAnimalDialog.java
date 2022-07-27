
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JSlider;

/**
 * DuplicateAnimalDialog which is built to make a duplicate swimmable
 * @author oReL
 */
public class DuplicateAnimalDialog extends JDialog implements ActionListener
{ 
	private JComboBox<String> cbcolors;
	private JButton submit;
	private JSlider size;
	private JSlider jhspeed;
	private JSlider jvspeed;
	private JSlider frequency;
	private static String colordecided;
	private AquaPanel ap;
	private int index;
	/**
	 * c'tor which is make a new dialog to build the swimmable
	 
	 * @param a as a callback from aquapanel
	 */
	public DuplicateAnimalDialog(AquaPanel a,int index)
	{
		super();
		ap=a;
		colordecided="black";
		this.index=index;
		size = new JSlider(JSlider.HORIZONTAL, 20, 320, 170);
		size.setMinorTickSpacing(0);
		size.setMajorTickSpacing(60);
		size.setPaintTicks(true);
		size.setPaintLabels(true);
		jhspeed = new JSlider(JSlider.HORIZONTAL, 1, 10, 5);
		jhspeed.setMinorTickSpacing(0);
		jhspeed.setMajorTickSpacing(1);
		jhspeed.setPaintTicks(true);
		jhspeed.setPaintLabels(true);
		jvspeed = new JSlider(JSlider.HORIZONTAL, 1, 10, 5);
		jvspeed.setMinorTickSpacing(0);
		jvspeed.setMajorTickSpacing(1);
		jvspeed.setPaintTicks(true);
		jvspeed.setPaintLabels(true);
		frequency = new JSlider(JSlider.HORIZONTAL, 50, 100, 75);
		frequency.setMinorTickSpacing(0);
		frequency.setMajorTickSpacing(5);
		frequency.setPaintTicks(true);
		frequency.setPaintLabels(true);
		String[] colors={"black","blue","red","yellow","magenta","orange","Same Color"};
		cbcolors=new JComboBox<String>(colors);
		cbcolors.addActionListener(this);
		submit=new JButton("Submit");
		submit.addActionListener(this);
		this.setLayout(new GridLayout(6,2));
		this.add(new JLabel("Select Size:"));
		this.add(size);
		this.add(new JLabel("Select HSpeed:"));
		this.add(jhspeed);
		this.add(new JLabel("Select VSpeed:"));
		this.add(jvspeed);
		this.add(new JLabel("Select Color:"));
		this.add(cbcolors);
		this.add(new JLabel("Food Frequency(in moves):"));
		this.add(frequency);
		this.add(new JLabel("To Finish click:"));
		this.add(submit);
		this.pack();
		this.setVisible(true);
	}
	/**
	 * function that make the new swimmable after the info chosen in the dailog before
	 */
	public void MakeDialog()
	{
		int hspeed=0;
		int vspeed=0;
		int sized=0;
		int freq;
		sized=size.getValue();
		hspeed=jhspeed.getValue();
		vspeed=jvspeed.getValue();
		freq=frequency.getValue();
		ArrayList<Swimmable> ar=ap.getSwimmableArray();
		Color color;
		if(DuplicateAnimalDialog.colordecided=="blue")
		{
			color=Color.blue;
		}
		else if(DuplicateAnimalDialog.colordecided=="red")
		{
			color=Color.red;
		}
		else if(DuplicateAnimalDialog.colordecided=="yellow")
		{
			color=Color.yellow;
		}
		else if(DuplicateAnimalDialog.colordecided=="magenta")
		{
			color=Color.magenta;
		}
		else if(DuplicateAnimalDialog.colordecided=="orange")
		{
			color=Color.orange;
		}
		else if(DuplicateAnimalDialog.colordecided=="black")
		{
			color=Color.black;
		}
		else
			color=ar.get(index).getColor();
		Swimmable s=ar.get(index).clone();
		s.UpdateDuplicateAttributes(hspeed,vspeed,color,sized);
		ap.addSwimmable(s);
		dispose();
		
	}
	/**
	 * actionperformed from actionlistener interface,to implement all the functions of the dailog when clicking on the buttons/comboboxes
	 *@para Actionevent e
	 *@see MakeDialog()
	 */
	@Override
	public void actionPerformed(ActionEvent e) 
	{
		if(e.getSource()==cbcolors)
		{
			DuplicateAnimalDialog.colordecided=(String)cbcolors.getSelectedItem();
		}
		else if(e.getSource()==submit)
		{
			MakeDialog();
		}
		
	}
	
}
