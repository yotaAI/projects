# projects
<!-- Install SSH in WSL -->
Status of SSH : sudo systemctl status ssh
Install SSH : 
`bash
sudo apt-get install openssh-server
sudo systemctl enable ssh --now
`
<!-- Get IP -->
`bash
ip addr | grep eth0
`
--------------------------
# Run Automantically
using cron

`bash
sudo apt install cron
crontab -e
Example : 0 9 * * * /home/user/my-agent/run_agent.sh >> /home/user/logs/agent.log 2>&1
`