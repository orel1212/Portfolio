
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FileDialog;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
/**
 * class that represent the frame of the aquarium
 * @author oReL
 *
 */

public class AquaFrame extends JFrame implements ActionListener
{
	private AquaPanel ap;
	private JMenuBar menu;
	private JMenu file;
	private JMenu bg;
	private JMenu help;
	private JMenuItem exit;
	private JMenuItem image;
	private JMenuItem blue;
	private JMenuItem none;
	private JMenuItem helpitem;
	private JMenu Memento;
	private JMenuItem saveobjectstate;
	private JMenuItem restoreobjectstate;
	private JPanel savestatepanel;
	private JPanel restorestatepanel;
	private ArrayList<Memento> mementoarr;
	private static AquaFrame af;
	/**
	 * c'tor that make a new aquarium with menu and aquapanel
	 */
	public AquaFrame()
	{
		super("my Aquarium");
		this.setSize(700, 700);
		af=this;
		mementoarr=new ArrayList<Memento>(5);
		ap=new AquaPanel(this);
		menu=new JMenuBar();
		file=new JMenu("File");
		file.addActionListener(this);
		bg=new JMenu("Background");
		bg.addActionListener(this);
		help=new JMenu("Help");
		help.addActionListener(this);
		Memento=new JMenu("Memento");
		Memento.addActionListener(this);
		menu.add(file);
		menu.add(bg);
		menu.add(Memento);
		menu.add(help);
		this.setJMenuBar(menu);
		helpitem=new JMenuItem("Help");
		image=new JMenuItem("Image");
		blue=new JMenuItem("Blue");
		none=new JMenuItem("None");
		saveobjectstate=new JMenuItem("Save Object State");
		restoreobjectstate=new JMenuItem("Restore Object State");
		exit=new JMenuItem("Exit");
		saveobjectstate.addActionListener(this);
		restoreobjectstate.addActionListener(this);
		none.addActionListener(this);
		blue.addActionListener(this);
		image.addActionListener(this);
		helpitem.addActionListener(this);
		exit.addActionListener(this);
		file.add(exit);
		Memento.add(saveobjectstate);
		Memento.addSeparator();
		Memento.add(restoreobjectstate);
		bg.add(image);
		bg.addSeparator();
		bg.add(blue);
		bg.addSeparator();
		bg.add(none);
		help.add(helpitem);
		this.add(ap);
		savestatepanel=null;
		restorestatepanel=null;
		
	}
	public static void main(String[] args) 
	{
		AquaFrame af = new AquaFrame();
		
	    // Exit when the window is closed.
	    af.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 
		
		// Show the converter.
		af.setResizable(false);
		af.setVisible(true);
		
	}
	@Override
	/**
	 * actionperformed from actionlistener interface,to implement all the functions of the menu when clicking on the buttons
	 *@para Actionevent e
	 */
	public void actionPerformed(ActionEvent e) 
	{
		if(e.getSource()==exit)
		{
			System.exit(0);
		}
		else if(e.getSource()==image)
		{
			FileDialog fd=new FileDialog(this,"Choose file",FileDialog.LOAD);
			fd.setVisible(true);
			String path=fd.getDirectory()+fd.getFile();
			ap.setBackgroundImage(path);
		}
		else if(e.getSource()==blue)
		{
			ap.setBackgroundColor("Blue");
		}
		else if(e.getSource()==none)
		{
			ap.setBackgroundColor("Default");
		}
		else if(e.getSource()==helpitem)
		{
			JOptionPane.showMessageDialog(null,"Home Work 3\n GUI @ Threads");
		}
		else if(e.getSource()==saveobjectstate)
		{
			ArrayList<Swimmable> swimtemp=ap.getSwimmableArray();
			ArrayList<Immobile> immotemp=ap.getImmobileArray();
			if(swimtemp.size()==0 && immotemp.size()==0)
				JOptionPane.showMessageDialog(null,"You cannot do save if there is not at least one SeaCreature!", "Error",JOptionPane.ERROR_MESSAGE);
			else
			{
				savestatepanel=new JPanel();
				savestatepanel.setLayout(new BorderLayout());
				JPanel savepanel=new JPanel();
				savepanel.setLayout(new FlowLayout());
				JButton savecreature=new JButton("Save");
				JLabel jl=new JLabel("Enter the index:");
				JTextField indexfield=new JTextField("",20);
				String [] columns={"Index","SeaCreature","Color","Size","Hor. speed","Ver. speed"};
				String[][] rows=new String[swimtemp.size()+immotemp.size()][6];
				for(int i=0;i<swimtemp.size();i++)
				{
					rows[i][0]=""+i;
					rows[i][1]=swimtemp.get(i).getAnimalName();
					Color c=swimtemp.get(i).getColor();
					if(c==Color.blue)
					{
						rows[i][2]="Blue";
					}
					else if(c==Color.red)
					{
						rows[i][2]="Red";
					}
					else if(c==Color.yellow)
					{
						rows[i][2]="Yellow";
					}
					else if(c==Color.magenta)
					{
						rows[i][2]="Magenta";
					}
					else if(c==Color.orange)
					{
						rows[i][2]="Orange";
					}
					else if(c==Color.black)
					{
						rows[i][2]="Black";
					}
					else
						rows[i][2]="("+c.getRed()+","+c.getGreen()+","+c.getBlue()+")";
					rows[i][3]=""+swimtemp.get(i).getSize();
					rows[i][4]=""+swimtemp.get(i).getHorSpeed();
					rows[i][5]=""+swimtemp.get(i).getVerSpeed();
					
				}
				for(int i=swimtemp.size();i<immotemp.size()+swimtemp.size();i++)
				{
					rows[i][0]=""+i;
					rows[i][1]=immotemp.get(i-swimtemp.size()).getPlantName();
					rows[i][2]="Green";
					rows[i][3]=""+immotemp.get(i-swimtemp.size()).getSize();
					rows[i][4]="0";
					rows[i][5]="0";
					
				}
				JTable jt=new JTable(rows,columns);
				JScrollPane js=new JScrollPane(jt);
				savecreature.addActionListener(new ActionListener()
				{
					@Override
					public void actionPerformed(ActionEvent e) 
					{
						if(AquaPanel.isInteger(indexfield.getText()))
						{
							int index=Integer.parseInt(indexfield.getText());
							
							if(index<swimtemp.size() && index>=0)
							{
								boolean checkifexist=false;
								for(int i=0;i<mementoarr.size();i++)
								{
									if(mementoarr.get(i).getState()==swimtemp.get(index).getSeaCreatureState())
									{
										checkifexist=true;
										mementoarr.get(i).getState().UpdateInfo(swimtemp.get(index).getInfo());
										break;
									}
								}
								if(!checkifexist)
								{
									Memento m=swimtemp.get(index).MakeMomento();
									m.getState().setIndex(index);
									mementoarr.add(mementoarr.size(),m);
								}
								savestatepanel.setVisible(false);
								af.remove(savestatepanel);
								af.add(ap);
								ap.setVisible(true);
							}
							else if(index<swimtemp.size()+immotemp.size() && index>=0)
							{
								boolean checkifexist=false;
								for(int i=0;i<mementoarr.size();i++)
								{
									if(mementoarr.get(i).getState()==immotemp.get(index-swimtemp.size()).getSeaCreatureState())
									{
										checkifexist=true;
										mementoarr.get(i).getState().UpdateInfo(immotemp.get(index-swimtemp.size()).getInfo());
										break;
									}
								}
								if(!checkifexist)
								{
									Memento m=immotemp.get(index-swimtemp.size()).MakeMomento();
									m.getState().setIndex(index-swimtemp.size());
									mementoarr.add(mementoarr.size(),m);
								}
								savestatepanel.setVisible(false);
								af.remove(savestatepanel);
								af.add(ap);
								ap.setVisible(true);
							}
							else
							{
								JOptionPane.showMessageDialog(null ,"You insert index that not in the range!","Error",JOptionPane.ERROR_MESSAGE);
								indexfield.setText("");
							}
						}
						else
						{
							JOptionPane.showMessageDialog(null,"You insert some string that not represent int!", "Error",JOptionPane.ERROR_MESSAGE);
							indexfield.setText("");
						}
						
					}
				});
				ap.setVisible(false);
				this.remove(ap);
				savestatepanel.add(js);
				savestatepanel.setVisible(false);
				js.setVisible(true);
				repaint();
				savepanel.add(jl);
				savepanel.add(indexfield);
				savepanel.add(savecreature);
				savestatepanel.add(savepanel,BorderLayout.SOUTH);
				this.add(savestatepanel);
				savestatepanel.setVisible(true);
			}
		}
		else if(e.getSource()==restoreobjectstate)
		{
			int totalsize=mementoarr.size();
			if(totalsize>0)
			{
				restorestatepanel=new JPanel();
				restorestatepanel.setLayout(new BorderLayout());
				JPanel restorepanel=new JPanel();
				restorepanel.setLayout(new FlowLayout());
				JButton restorecreature=new JButton("Restore");
				JLabel jl=new JLabel("Enter the index:");
				JTextField indexfield=new JTextField("",20);
				String [] columns={"Index","SeaCreature","Color","Size","x","y","Hor. speed","Ver. speed"};
				
				String[][] rows=new String[totalsize][8];
				
				for(int i=0;i<mementoarr.size();i++)
				{
					rows[i][0]=""+i;
					rows[i][1]=mementoarr.get(i).getState().getName();
					Color c=mementoarr.get(i).getState().getCol();
					if(c==Color.blue)
					{
						rows[i][2]="Blue";
					}
					else if(c==Color.red)
					{
						rows[i][2]="Red";
					}
					else if(c==Color.yellow)
					{
						rows[i][2]="Yellow";
					}
					else if(c==Color.magenta)
					{
						rows[i][2]="Magenta";
					}
					else if(c==Color.orange)
					{
						rows[i][2]="Orange";
					}
					else if(c==Color.black)
					{
						rows[i][2]="Black";
					}
					else if(c==Color.green)
					{
						rows[i][2]="Green";
					}
					else
						rows[i][2]="("+c.getRed()+","+c.getGreen()+","+c.getBlue()+")";
					rows[i][3]=""+mementoarr.get(i).getState().getSize();
					rows[i][4]=""+mementoarr.get(i).getState().getX();
					rows[i][5]=""+mementoarr.get(i).getState().getY();
					rows[i][6]=""+mementoarr.get(i).getState().getHorspeed();
					rows[i][7]=""+mementoarr.get(i).getState().getVerspeed();
				}
				JTable jt=new JTable(rows,columns);
				JScrollPane js=new JScrollPane(jt);
				restorecreature.addActionListener(new ActionListener()
				{
					@Override
					public void actionPerformed(ActionEvent e) 
					{
						if(AquaPanel.isInteger(indexfield.getText()))
						{
							int index=Integer.parseInt(indexfield.getText());
							
							
							if(index>=0 && index<mementoarr.size())
							{
								int seaindex=mementoarr.get(index).getState().getIndex();
								if(mementoarr.get(index).getState().getName().equals("Fish") || mementoarr.get(index).getState().getName().equals("Jellyfish"))
								{
									ArrayList<Swimmable> swimtemp=ap.getSwimmableArray();
									swimtemp.get(seaindex).setInfo(mementoarr.get(index).getState().getInfo());
								}
								else
								{
									ArrayList<Immobile> immotemp=ap.getImmobileArray();
									immotemp.get(seaindex).setInfo(mementoarr.get(index).getState().getInfo());
								}
								restorestatepanel.setVisible(false);
								af.remove(restorestatepanel);
								af.add(ap);
								ap.setVisible(true);
								
							}
							else
							{
								JOptionPane.showMessageDialog(null ,"You insert index that not in the range!","Error",JOptionPane.ERROR_MESSAGE);
								indexfield.setText("");
							}
						}
						else
						{
							JOptionPane.showMessageDialog(null,"You insert some string that not represent int!", "Error",JOptionPane.ERROR_MESSAGE);
							indexfield.setText("");
						}
						
					}
				});
				ap.setVisible(false);
				this.remove(ap);
				restorestatepanel.add(js);
				restorestatepanel.setVisible(false);
				js.setVisible(true);
				repaint();
				restorepanel.add(jl);
				restorepanel.add(indexfield);
				restorepanel.add(restorecreature);
				restorestatepanel.add(restorepanel,BorderLayout.SOUTH);
				restorestatepanel.setVisible(true);
				this.add(restorestatepanel);
			}
			else
			{
				JOptionPane.showMessageDialog(null,"There are no saved states to restore!", "Error",JOptionPane.ERROR_MESSAGE);
			}
				
		}
	}
	/**
	 * function that removes all the stats that saved
	 */
	public void ResetSavedStates()
	{
		mementoarr.clear();
	}
	
}
